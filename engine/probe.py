"""Device capability probe.

Every worker runs this once at startup and reports the result to the
coordinator. The coordinator's admission gate and shard planner need a number
that is comparable across NVIDIA, Intel and CPU devices, and vendor-reported
specs are not comparable. So we measure the one thing that actually predicts
YOLO step time: sustained fused-multiply-add throughput at a realistic tile
size, plus the memory ceiling that bounds batch size.

The probe is deliberately short (a few hundred milliseconds). It is a ranking
signal, not a benchmark suite.
"""

from __future__ import annotations

import platform
import socket
import time
from dataclasses import asdict, dataclass
from typing import Optional

from accelerator import Accelerator


@dataclass(frozen=True)
class Capability:
    backend: str
    device_name: str
    total_memory_mb: int
    gflops: float
    mem_bandwidth_gbps: float
    host: str
    platform: str
    torch_version: str
    supports_training: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _timed_matmul(torch, accelerator: Accelerator, size: int, iters: int) -> float:
    """Return achieved GFLOP/s for an size x size FP32 matmul."""
    device = accelerator.torch_device
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)

    # Warm up: first call pays kernel autotuning and allocator cost.
    for _ in range(3):
        a @ b
    accelerator.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    accelerator.synchronize()
    elapsed = time.perf_counter() - start

    if elapsed <= 0:
        return 0.0
    flops = 2.0 * (size ** 3) * iters
    return flops / elapsed / 1e9


def _timed_copy(torch, accelerator: Accelerator, mb: int, iters: int) -> float:
    """Return achieved device memory bandwidth in GB/s."""
    device = accelerator.torch_device
    elements = (mb * 1024 * 1024) // 4
    src = torch.empty(elements, device=device, dtype=torch.float32).fill_(1.0)
    dst = torch.empty_like(src)

    dst.copy_(src)
    accelerator.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    accelerator.synchronize()
    elapsed = time.perf_counter() - start

    if elapsed <= 0:
        return 0.0
    moved_bytes = 2.0 * elements * 4 * iters  # one read plus one write
    return moved_bytes / elapsed / 1e9


def probe(accelerator: Accelerator) -> Capability:
    import torch

    # CPU devices get a smaller problem so the probe stays sub-second there too.
    if accelerator.backend == "cpu":
        size, iters, mb, copies = 512, 6, 64, 6
    else:
        size, iters, mb, copies = 2048, 12, 256, 12

    try:
        gflops = _timed_matmul(torch, accelerator, size, iters)
    except Exception:
        gflops = 0.0
    try:
        bandwidth = _timed_copy(torch, accelerator, mb, copies)
    except Exception:
        bandwidth = 0.0

    return Capability(
        backend=accelerator.backend,
        device_name=accelerator.device_name,
        total_memory_mb=accelerator.total_memory_mb,
        gflops=round(gflops, 2),
        mem_bandwidth_gbps=round(bandwidth, 2),
        host=socket.gethostname(),
        platform=f"{platform.system()} {platform.release()}",
        torch_version=getattr(torch, "__version__", "unknown"),
        supports_training=accelerator.supports_training,
    )


def probe_or_none(accelerator: Optional[Accelerator]) -> Optional[dict]:
    if accelerator is None:
        return None
    try:
        return probe(accelerator).as_dict()
    except Exception:
        return None


if __name__ == "__main__":
    from accelerator import detect_accelerator

    result = probe(detect_accelerator("auto"))
    for key, value in result.as_dict().items():
        print(f"{key:20} {value}")
