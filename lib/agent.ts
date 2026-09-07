/**
 * The exact set of files a joining machine downloads to become a worker.
 *
 * An explicit allowlist rather than a directory read: the route that serves
 * these is reachable by anyone who can open the join page, and it must not
 * become a way to read the host's filesystem.
 */
export const AGENT_FILES = [
  "accelerator.py",
  "federated_training.py",
  "ultralytics_xpu.py",
  "probe.py",
  "worker.py",
  "requirements-control.txt",
  "requirements-train.txt",
  "requirements-train-cpu.txt",
  "requirements-xpu.txt",
] as const;

export const AGENT_FILE_SET = new Set<string>(AGENT_FILES);
