import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import net from "node:net";
import { networkInterfaces } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
export const ENGINE_DIR = path.join(REPO_ROOT, "engine");
export const STATE_DIR = path.join(REPO_ROOT, ".gradmesh");
export const VENV_DIR = path.join(REPO_ROOT, ".venv");
export const IS_WINDOWS = process.platform === "win32";

export const COORDINATOR_PORT = Number(process.env.GRADMESH_COORDINATOR_PORT || 8000);
export const WEB_PORT = Number(process.env.PORT || 3000);

const RESET = "\u001b[0m";
const COLORS = {
  gray: "\u001b[90m",
  cyan: "\u001b[36m",
  green: "\u001b[32m",
  yellow: "\u001b[33m",
  red: "\u001b[31m",
  magenta: "\u001b[35m",
  bold: "\u001b[1m",
};

export function paint(color, text) {
  if (!process.stdout.isTTY) return text;
  return `${COLORS[color] || ""}${text}${RESET}`;
}

export function log(scope, message, color = "cyan") {
  console.log(`${paint(color, scope.padEnd(11))} ${message}`);
}

export function banner(lines) {
  const width = Math.max(...lines.map((line) => stripAnsi(line).length)) + 2;
  const bar = "─".repeat(width);
  console.log(paint("gray", `╭${bar}╮`));
  for (const line of lines) {
    const pad = " ".repeat(width - stripAnsi(line).length - 1);
    console.log(`${paint("gray", "│")} ${line}${pad}${paint("gray", "│")}`);
  }
  console.log(paint("gray", `╰${bar}╯`));
}

function stripAnsi(value) {
  // eslint-disable-next-line no-control-regex
  return value.replace(new RegExp(String.fromCharCode(27) + "\\[[0-9;]*m", "g"), "");
}

export function venvPython() {
  return IS_WINDOWS
    ? path.join(VENV_DIR, "Scripts", "python.exe")
    : path.join(VENV_DIR, "bin", "python");
}

export function venvReady() {
  return existsSync(venvPython());
}

/**
 * Find a usable system Python. Windows ships a `python` shim that opens the
 * Microsoft Store instead of running anything, so a version check is the only
 * reliable probe.
 */
export function findSystemPython() {
  const candidates = IS_WINDOWS
    ? ["py -3.12", "py -3.11", "py -3", "python", "python3"]
    : ["python3.12", "python3.11", "python3", "python"];

  for (const candidate of candidates) {
    const [command, ...args] = candidate.split(" ");
    // No shell here: cmd.exe mangles the quoting, and PATH lookup for .exe
    // files works without one anyway.
    const probe = spawnSync(
      command,
      [...args, "-c", "import sys;print(sys.version_info.major);print(sys.version_info.minor)"],
      { encoding: "utf8" }
    );
    if (probe.status !== 0) continue;
    const [major, minor] = (probe.stdout || "").trim().split(/\s+/).map(Number);
    if (major === 3 && minor >= 9) {
      return { command, args, version: `${major}.${minor}` };
    }
  }
  return null;
}

export function ensureStateDir() {
  if (!existsSync(STATE_DIR)) mkdirSync(STATE_DIR, { recursive: true });
  return STATE_DIR;
}

export function readCoordinatorState() {
  const file = path.join(STATE_DIR, "coordinator.json");
  if (!existsSync(file)) return null;
  try {
    return JSON.parse(readFileSync(file, "utf8"));
  } catch {
    return null;
  }
}

/** Every routable IPv4 address this host answers on, best guess first. */
export function lanAddresses() {
  const found = [];
  for (const [name, entries] of Object.entries(networkInterfaces())) {
    for (const entry of entries || []) {
      if (entry.family !== "IPv4" || entry.internal) continue;
      found.push({ name, address: entry.address });
    }
  }
  // Prefer private ranges, since that is what a peer on the same Wi-Fi will use.
  const score = (address) =>
    address.startsWith("192.168.") ? 0 : address.startsWith("10.") ? 1 : address.startsWith("172.") ? 2 : 3;
  found.sort((a, b) => score(a.address) - score(b.address));
  return found;
}

export function primaryLanAddress() {
  return lanAddresses()[0]?.address || "127.0.0.1";
}

export function run(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd || REPO_ROOT,
      env: { ...process.env, ...(options.env || {}) },
      stdio: options.stdio || "inherit",
      shell: options.shell ?? false,
    });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0 || options.allowFailure) resolve(code);
      else reject(new Error(`${command} exited with code ${code}`));
    });
  });
}

export function spawnBackground(command, args, options = {}) {
  const child = spawn(command, args, {
    cwd: options.cwd || REPO_ROOT,
    env: { ...process.env, ...(options.env || {}) },
    stdio: options.stdio || ["ignore", "pipe", "pipe"],
    shell: options.shell ?? false,
  });
  if (options.label && child.stdout) {
    pipeLines(child.stdout, options.label, options.color || "gray", options.filter);
  }
  if (options.label && child.stderr) {
    pipeLines(child.stderr, options.label, options.color || "gray", options.filter);
  }
  return child;
}

function pipeLines(stream, label, color, filter) {
  let buffer = "";
  stream.setEncoding("utf8");
  stream.on("data", (chunk) => {
    buffer += chunk;
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      if (filter && !filter(line)) continue;
      console.log(`${paint(color, label.padEnd(11))} ${line}`);
    }
  });
}

/**
 * Is something already listening here?
 *
 * Worth checking before spawning anything: a stale dev server holding port 3000
 * otherwise surfaces as a raw EADDRINUSE stack trace from deep inside Next,
 * which tells the user nothing about what to do next.
 */
export function portInUse(port, host = "127.0.0.1") {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    const done = (result) => {
      socket.destroy();
      resolve(result);
    };
    socket.setTimeout(700);
    socket.once("connect", () => done(true));
    socket.once("timeout", () => done(false));
    socket.once("error", () => done(false));
    socket.connect(port, host);
  });
}

export async function waitForHttp(url, { timeoutMs = 120000, intervalMs = 400 } = {}) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(2500) });
      if (response.ok) return await response.json().catch(() => ({}));
    } catch {
      // not up yet
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  throw new Error(`Timed out waiting for ${url}`);
}

/**
 * Which PyTorch build this machine needs.
 *
 * Matching on the vendor name alone is not enough, and getting that wrong is
 * silent until training starts. A CUDA wheel only contains compiled kernels for
 * the architectures it was built against: the cu12.1 build covers sm_50 through
 * sm_90, which is Maxwell through Hopper. Blackwell, the RTX 50 series, is
 * sm_120 and appears in none of them. Such a card installs cleanly, reports
 * itself as available, and then fails every kernel launch with "no kernel image
 * is available for execution on the device". No driver update fixes it, because
 * the kernels were never compiled.
 *
 * nvidia-smi reports the compute capability without PyTorch being installed,
 * which resolves the chicken and egg, and is why the profile is chosen from a
 * number rather than from a marketing name.
 */
export function detectGpuProfile() {
  const nvidia = queryNvidiaSmi();
  if (nvidia) {
    const { name, capability } = nvidia;
    if (capability === null) {
      // An old nvidia-smi with no compute_cap field. cu121 covers everything
      // that shipped before that field existed, so it is the safer guess.
      return {
        vendor: "cuda",
        name,
        capability: null,
        profile: "cuda-cu121",
        file: "requirements-train-cu121.txt",
        label: "PyTorch CUDA 12.1 build",
        backend: "cuda",
        reason: "nvidia-smi did not report a compute capability, assuming pre-Blackwell",
      };
    }
    if (capability >= 12.0) {
      return {
        vendor: "cuda",
        name,
        capability,
        profile: "cuda-cu128",
        file: "requirements-train-cu128.txt",
        label: "PyTorch CUDA 12.8 build",
        backend: "cuda",
        reason: `compute capability ${capability.toFixed(1)} is Blackwell or newer, which needs CUDA 12.8`,
      };
    }
    if (capability >= 5.0) {
      return {
        vendor: "cuda",
        name,
        capability,
        profile: "cuda-cu121",
        file: "requirements-train-cu121.txt",
        label: "PyTorch CUDA 12.1 build",
        backend: "cuda",
        reason: `compute capability ${capability.toFixed(1)} is covered by the CUDA 12.1 build`,
      };
    }
    return {
      vendor: "cpu",
      name,
      capability,
      profile: "cpu",
      file: "requirements-train-cpu.txt",
      label: "PyTorch CPU build",
      backend: "cpu",
      reason: `compute capability ${capability.toFixed(1)} is too old for any current PyTorch CUDA build`,
    };
  }

  const names = videoControllerNames().toLowerCase();
  if (names.includes("intel(r) arc") || names.includes("intel arc") || names.includes("intel corporation dg2")) {
    return {
      vendor: "xpu",
      name: "Intel Arc",
      capability: null,
      profile: "xpu",
      file: "requirements-xpu.txt",
      label: "PyTorch Intel XPU build",
      backend: "xpu",
      reason: "an Intel Arc GPU was detected",
    };
  }

  return {
    vendor: "cpu",
    name: names.split(";")[0] || "unknown",
    capability: null,
    profile: "cpu",
    file: "requirements-train-cpu.txt",
    label: "PyTorch CPU build",
    backend: "cpu",
    reason: "no supported GPU was detected",
  };
}

/** Name and compute capability from nvidia-smi, or null when there is no NVIDIA GPU. */
function queryNvidiaSmi() {
  try {
    const probe = spawnSync(
      "nvidia-smi",
      ["--query-gpu=name,compute_cap", "--format=csv,noheader"],
      { encoding: "utf8" }
    );
    if (probe.status !== 0 || !probe.stdout) return null;

    // Multiple GPUs: take the most capable, which is what training will use.
    const rows = probe.stdout
      .trim()
      .split(/\r?\n/)
      .map((line) => line.split(",").map((part) => part.trim()))
      .filter((parts) => parts[0]);
    if (rows.length === 0) return null;

    let best = { name: rows[0][0], capability: null };
    for (const [name, cap] of rows) {
      const parsed = Number.parseFloat(cap);
      const capability = Number.isFinite(parsed) ? parsed : null;
      if (capability !== null && (best.capability === null || capability > best.capability)) {
        best = { name, capability };
      }
    }
    return best;
  } catch {
    return null;
  }
}

function videoControllerNames() {
  try {
    if (IS_WINDOWS) {
      const probe = spawnSync(
        "powershell",
        ["-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name -join ';'"],
        { encoding: "utf8" }
      );
      return probe.stdout || "";
    }
    if (process.platform === "linux") {
      const probe = spawnSync("sh", ["-c", "lspci 2>/dev/null | grep -i 'vga\\|3d\\|display'"], {
        encoding: "utf8",
      });
      return probe.stdout || "";
    }
  } catch {
    // Fall through to unknown.
  }
  return "";
}

/** Kept for callers that only need the coarse answer. */
export function detectGpuVendor() {
  return detectGpuProfile().vendor;
}
