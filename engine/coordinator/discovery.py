"""Finding things on the local network, in both directions.

Two problems, and they are not the same problem.

**A device needs to find the host.** Telling someone "open http://192.168.1.85:3000"
fails the moment the router hands out a new lease, and nobody wants to read an
IP address off a screen. So the coordinator advertises itself over multicast DNS
and claims the name `gradmesh.local`. Any device on the network with an mDNS
resolver, which is Windows 10 and later, macOS, iOS, Android 12 and later and
most desktop Linux, can then just open `http://gradmesh.local:3000` with no
setup on either side.

**The host needs to see what is out there.** `scan_network()` sweeps the local
subnet so the dashboard can show every device on the Wi-Fi, which of them are
already contributing, and which are sitting there with a GPU doing nothing.

Liveness detection uses a detail worth stating: a TCP connection that is
*refused* proves a host exists just as well as one that is accepted. Only a
timeout means nothing is there. So probing a couple of common ports finds
devices that answer nothing at all, which is most phones and laptops.
"""

from __future__ import annotations

import errno
import json
import re
import selectors
import socket
import subprocess
import sys
import time
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence
from urllib import request

# Ports that would make a device interesting to GradMesh specifically.
MESH_PORTS = (8000, 3000)
# Ports probed to decide whether an address is a live device at all. Kept short
# on purpose: every extra port multiplies the work by the size of the subnet.
LIVENESS_PORTS = (80, 443, 445, 22)
PROBE_PORTS = MESH_PORTS + LIVENESS_PORTS

_REFUSED_ERRNOS = {errno.ECONNREFUSED, errno.ECONNRESET, 10061, 10054}

SCAN_TIMEOUT = 0.4
# Sockets in flight at once. Windows' select() tops out around 512 descriptors,
# so batches stay well under that.
SCAN_BATCH = 384


@dataclass
class Device:
    ip: str
    mac: Optional[str] = None
    hostname: Optional[str] = None
    open_ports: List[int] = field(default_factory=list)
    is_coordinator: bool = False
    is_this_host: bool = False
    source: str = "scan"

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Addressing
# ---------------------------------------------------------------------------


def local_ipv4() -> str:
    """The address this host uses to reach the LAN, without sending traffic."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))
        return probe.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        probe.close()


def subnet_hosts(base: Optional[str] = None) -> List[str]:
    """Every address in the local /24, excluding the network and broadcast."""
    base = base or local_ipv4()
    if base.startswith("127."):
        return []
    prefix = base.rsplit(".", 1)[0]
    return ["%s.%d" % (prefix, octet) for octet in range(1, 255)]


# ---------------------------------------------------------------------------
# ARP
# ---------------------------------------------------------------------------

_ARP_WINDOWS = re.compile(r"^\s*(\d+\.\d+\.\d+\.\d+)\s+([0-9a-fA-F-]{17})\s+(\w+)")
_ARP_UNIX = re.compile(r"\(?(\d+\.\d+\.\d+\.\d+)\)?\s+at\s+([0-9a-fA-F:]{11,17})")


def arp_table() -> Dict[str, str]:
    """Map IP to MAC for devices this host has recently exchanged frames with.

    Free information: the OS already has it, so it costs one subprocess call and
    it names devices a port scan would miss entirely.
    """
    try:
        result = subprocess.run(
            ["arp", "-a"],
            capture_output=True,
            text=True,
            timeout=6,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0,
        )
    except Exception:
        return {}

    table: Dict[str, str] = {}
    for line in (result.stdout or "").splitlines():
        match = _ARP_WINDOWS.match(line) or _ARP_UNIX.search(line)
        if not match:
            continue
        ip, mac = match.group(1), match.group(2)
        if mac.lower() in {"ff-ff-ff-ff-ff-ff", "ff:ff:ff:ff:ff:ff"}:
            continue
        if ip.endswith(".255") or ip.startswith("224.") or ip.startswith("239."):
            continue
        table[ip] = mac.replace("-", ":").lower()
    return table


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


def _port_state(ip: str, port: int, timeout: float = SCAN_TIMEOUT) -> str:
    """One of open, refused or closed. Refused still proves the host exists."""
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.settimeout(timeout)
    try:
        connection.connect((ip, port))
        return "open"
    except ConnectionRefusedError:
        return "refused"
    except OSError as exc:
        if getattr(exc, "winerror", None) in {10054, 10061}:
            return "refused"
        return "closed"
    finally:
        connection.close()


def _sweep(targets: Sequence[tuple], timeout: float = SCAN_TIMEOUT) -> Dict[tuple, str]:
    """Probe many (host, port) pairs concurrently with non-blocking connects.

    A thread per socket does not work at this scale: 254 addresses times six
    ports is fifteen hundred connects, and the thread pool version spent seven
    seconds mostly on scheduling overhead. Non-blocking connects registered with
    a selector cost one file descriptor each and finish in about one timeout
    period regardless of how many there are.
    """
    results: Dict[tuple, str] = {}

    for start in range(0, len(targets), SCAN_BATCH):
        batch = targets[start : start + SCAN_BATCH]
        selector = selectors.DefaultSelector()
        sockets: Dict[int, tuple] = {}

        for target in batch:
            ip, port = target
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.setblocking(False)
            try:
                connection.connect_ex((ip, port))
                selector.register(connection, selectors.EVENT_WRITE, target)
                sockets[connection.fileno()] = (target, connection)
            except Exception:
                results[target] = "closed"
                connection.close()

        deadline = time.monotonic() + timeout
        while sockets and time.monotonic() < deadline:
            events = selector.select(max(0.01, deadline - time.monotonic()))
            for key, _mask in events:
                connection = key.fileobj
                target = key.data
                # SO_ERROR is how a non-blocking connect reports its outcome.
                error = connection.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)  # type: ignore[union-attr]
                if error == 0:
                    results[target] = "open"
                elif error in _REFUSED_ERRNOS:
                    results[target] = "refused"
                else:
                    results[target] = "closed"
                selector.unregister(connection)
                sockets.pop(connection.fileno(), None)  # type: ignore[union-attr]
                connection.close()  # type: ignore[union-attr]

        # Anything still pending never answered, so nothing is listening there.
        for target, connection in sockets.values():
            results.setdefault(target, "closed")
            try:
                selector.unregister(connection)
            except Exception:
                pass
            connection.close()
        selector.close()

    return results


def _reverse_dns(ip: str) -> Optional[str]:
    """Best-effort device name. Callers must bound this, see resolve_names()."""
    try:
        name = socket.gethostbyaddr(ip)[0]
        return name.split(".")[0] or None
    except Exception:
        return None


def resolve_names(ips: Sequence[str], budget: float = 1.5) -> Dict[str, str]:
    """Reverse-resolve as many addresses as fit in a time budget.

    gethostbyaddr goes through the system resolver, which on Windows ignores
    socket.setdefaulttimeout entirely: a single unresolvable address on a LAN
    blocked for five seconds and dominated the whole scan. There is no portable
    way to pass it a timeout, so the lookups run on daemon threads and whatever
    has not answered inside the budget is simply left unnamed. An unnamed device
    still shows its address and MAC, which is enough to recognise it.
    """
    if not ips:
        return {}

    names: Dict[str, str] = {}
    pool = ThreadPoolExecutor(max_workers=min(16, len(ips)))
    try:
        futures = {pool.submit(_reverse_dns, ip): ip for ip in ips}
        done, _pending = wait(futures, timeout=budget, return_when=ALL_COMPLETED)
        for future in done:
            try:
                name = future.result(timeout=0)
            except Exception:
                continue
            if name:
                names[futures[future]] = name
    finally:
        # Do not block shutdown on a resolver call that may never return.
        pool.shutdown(wait=False, cancel_futures=True)
    return names


def _is_gradmesh_coordinator(ip: str) -> bool:
    try:
        req = request.Request("http://%s:8000/health" % ip, method="GET")
        with request.urlopen(req, timeout=1.2) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return payload.get("status") == "ok" and "version" in payload
    except Exception:
        return False


def scan_network(base: Optional[str] = None) -> dict:
    """Sweep the local subnet. Roughly two seconds on a home network."""
    started = time.perf_counter()
    self_ip = local_ipv4()
    hosts = subnet_hosts(base or self_ip)
    macs = arp_table()

    if not hosts:
        return {
            "devices": [],
            "subnet": None,
            "self_ip": self_ip,
            "scanned": 0,
            "duration_seconds": 0.0,
            "at": time.time(),
        }

    targets = [(ip, port) for ip in hosts for port in PROBE_PORTS]
    open_ports: Dict[str, List[int]] = {}
    alive = {ip for ip in macs if ip in set(hosts)} | {self_ip}

    for (ip, port), state in _sweep(targets).items():
        if state == "open":
            open_ports.setdefault(ip, []).append(port)
            alive.add(ip)
        elif state == "refused":
            alive.add(ip)

    # Reverse DNS and the coordinator handshake only run for addresses that
    # actually answered, which is a handful rather than 254.
    found = sorted(alive, key=lambda item: tuple(int(part) for part in item.split(".")))

    names = resolve_names(found)

    # Only addresses with 8000 open get the coordinator handshake, so the
    # identify step costs one request rather than one per device.
    candidates = [ip for ip in found if 8000 in open_ports.get(ip, [])]
    with ThreadPoolExecutor(max_workers=max(1, len(candidates))) as pool:
        coordinators = set(
            ip for ip, ok in zip(candidates, pool.map(_is_gradmesh_coordinator, candidates)) if ok
        ) if candidates else set()

    devices = [
        Device(
            ip=ip,
            mac=macs.get(ip),
            hostname=names.get(ip),
            open_ports=sorted(set(open_ports.get(ip, []))),
            is_coordinator=ip in coordinators,
            is_this_host=ip == self_ip,
            source="arp" if ip in macs else "scan",
        )
        for ip in found
    ]
    return {
        "devices": [device.as_dict() for device in devices],
        "subnet": "%s.0/24" % self_ip.rsplit(".", 1)[0],
        "self_ip": self_ip,
        "scanned": len(hosts),
        "duration_seconds": round(time.perf_counter() - started, 2),
        "at": time.time(),
    }


# ---------------------------------------------------------------------------
# Multicast DNS
# ---------------------------------------------------------------------------


class MulticastAdvertiser:
    """Claims gradmesh.local so nobody has to type an IP address.

    Optional by design. zeroconf is a pure-Python dependency with no build step,
    but a locked-down network or a conflicting responder such as an existing
    Bonjour service should degrade to "use the IP" rather than stop the
    coordinator from starting.

    **start() and stop() must not be called from an asyncio event loop thread.**
    zeroconf runs its own loop on its own thread, and its synchronous API waits
    on that loop with run_coroutine_threadsafe. Called from inside uvicorn's
    loop that wait would deadlock, so zeroconf refuses with EventLoopBlocked
    instead. The coordinator therefore drives both through a thread executor.
    """

    def __init__(self, hostname: str = "gradmesh", web_port: int = 3000, api_port: int = 8000) -> None:
        self.hostname = hostname
        self.web_port = web_port
        self.api_port = api_port
        self._zeroconf = None
        self._services: List[object] = []
        self.error: Optional[str] = None
        self.address: Optional[str] = None

    @property
    def local_name(self) -> str:
        return "%s.local" % self.hostname

    @property
    def active(self) -> bool:
        return self._zeroconf is not None and not self.error

    def start(self) -> bool:
        try:
            from zeroconf import ServiceInfo, Zeroconf
        except Exception:
            self.error = "zeroconf is not installed, so gradmesh.local will not resolve"
            return False

        ip = local_ipv4()
        if ip.startswith("127."):
            self.error = "this host has no LAN address to advertise"
            return False

        try:
            packed = socket.inet_aton(ip)
            self._zeroconf = Zeroconf()
            self.address = ip

            # The dashboard is what a person opens, so it advertises the
            # hostname. The coordinator is advertised separately so a worker
            # agent can discover the control plane without guessing a port.
            self._services = [
                ServiceInfo(
                    "_http._tcp.local.",
                    "GradMesh._http._tcp.local.",
                    addresses=[packed],
                    port=self.web_port,
                    properties={"path": "/join", "role": "dashboard"},
                    server="%s.local." % self.hostname,
                ),
                ServiceInfo(
                    "_gradmesh._tcp.local.",
                    "GradMesh Coordinator._gradmesh._tcp.local.",
                    addresses=[packed],
                    port=self.api_port,
                    properties={"role": "coordinator", "version": "4.0.0"},
                    server="%s.local." % self.hostname,
                ),
            ]
            for service in self._services:
                self._zeroconf.register_service(service, allow_name_change=True)
            return True
        except Exception as exc:
            self.error = "%s: %s" % (type(exc).__name__, exc)
            self.stop()
            return False

    def stop(self) -> None:
        zeroconf = self._zeroconf
        self._zeroconf = None
        if zeroconf is None:
            return
        try:
            for service in self._services:
                try:
                    zeroconf.unregister_service(service)  # type: ignore[arg-type]
                except Exception:
                    pass
            zeroconf.close()
        except Exception:
            pass
        finally:
            self._services = []

    def as_dict(self) -> dict:
        return {
            "active": self.active,
            "hostname": self.local_name,
            "address": self.address,
            "web_port": self.web_port,
            "api_port": self.api_port,
            "error": self.error,
        }
