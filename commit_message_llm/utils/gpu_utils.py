"""GPU utility functions for monitoring and managing GPU memory."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GPUMemoryInfo:
    """Information about GPU memory usage."""

    device_id: int
    total_mb: int
    used_mb: int
    free_mb: int
    percent_used: float

    @property
    def total_gb(self) -> float:
        """Total memory in GB."""
        return self.total_mb / 1024

    @property
    def used_gb(self) -> float:
        """Used memory in GB."""
        return self.used_mb / 1024

    @property
    def free_gb(self) -> float:
        """Free memory in GB."""
        return self.free_mb / 1024

    def __str__(self) -> str:
        """String representation of GPU memory info."""
        return (
            f"GPU {self.device_id}: {self.used_mb} MB used / {self.total_mb} MB total "
            f"({self.percent_used:.1f}%)"
        )


def get_gpu_memory_info(device_id: int = 0) -> Optional[GPUMemoryInfo]:
    """
    Get GPU memory information for the specified device.

    Args:
        device_id: GPU device ID (default: 0).

    Returns:
        GPUMemoryInfo object with memory statistics, or None if GPU is not available.
    """
    if not torch.cuda.is_available():
        return None

    if device_id >= torch.cuda.device_count():
        return None

    try:
        props = torch.cuda.get_device_properties(device_id)
        total_mb = props.total_memory // (1024 * 1024)
        used_mb = torch.cuda.memory_allocated(device_id) // (1024 * 1024)
        free_mb = total_mb - used_mb
        percent_used = (used_mb / total_mb * 100) if total_mb > 0 else 0.0

        return GPUMemoryInfo(
            device_id=device_id,
            total_mb=total_mb,
            used_mb=used_mb,
            free_mb=free_mb,
            percent_used=percent_used,
        )
    except Exception:
        return None


def print_gpu_utilization(device_id: int = 0) -> None:
    """
    Print GPU memory utilization to the console.

    Args:
        device_id: GPU device ID (default: 0).
    """
    info = get_gpu_memory_info(device_id)
    if info is None:
        print("GPU not available or not found.")
    else:
        print(f"GPU memory occupied: {info.used_mb} MB.")


def get_all_gpu_memory_info() -> list[GPUMemoryInfo]:
    """
    Get GPU memory information for all available GPUs.

    Returns:
        List of GPUMemoryInfo objects for all available GPUs.
    """
    if not torch.cuda.is_available():
        return []

    gpu_count = torch.cuda.device_count()
    return [get_gpu_memory_info(i) for i in range(gpu_count)]


def clear_gpu_cache(device_id: Optional[int] = None) -> None:
    """
    Clear GPU cache to free memory.

    Args:
        device_id: Optional GPU device ID. If None, clears cache for all GPUs.
    """
    if not torch.cuda.is_available():
        return

    if device_id is not None:
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()
    else:
        torch.cuda.empty_cache()


def get_device(device_id: int = 0) -> torch.device:
    """
    Get the appropriate torch device (GPU if available, else CPU).

    Args:
        device_id: Preferred GPU device ID.

    Returns:
        torch.device object.
    """
    if torch.cuda.is_available() and device_id < torch.cuda.device_count():
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


__all__ = [
    "GPUMemoryInfo",
    "get_gpu_memory_info",
    "print_gpu_utilization",
    "get_all_gpu_memory_info",
    "clear_gpu_cache",
    "get_device",
]
