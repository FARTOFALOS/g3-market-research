"""Small live resource policy for safe local materialization."""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess
from ctypes import wintypes
from dataclasses import asdict, dataclass
from pathlib import Path

GIB = 1024 ** 3


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


@dataclass(frozen=True)
class ResourcePolicy:
    logical_cpus: int
    physical_cpus: int
    total_ram_bytes: int
    available_ram_bytes: int
    memory_load_percent: int
    reserved_ram_bytes: int
    build_ram_budget_bytes: int
    auto_worker_cap: int
    disk_free_bytes: int
    disk_reserve_bytes: int
    measured_tf1_peak_bytes: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def live_policy(repo_root: Path) -> ResourcePolicy:
    logical = os.cpu_count() or 1
    physical = max(1, logical // 2)
    if os.name == "nt":
        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        total, available = int(status.ullTotalPhys), int(status.ullAvailPhys)
        memory_load = int(status.dwMemoryLoad)
        try:
            raw = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum"],
                text=True, timeout=10,
            ).strip()
            physical = max(1, int(raw))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    else:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        total, available = pages * page_size, available_pages * page_size
        memory_load = round(100 * (1 - available / total))
    reserve = max(int(total * 0.22), 3 * GIB)
    # Do not claim currently occupied RAM merely because it exists in total.
    budget = max(512 * 1024 ** 2, min(int(total * 0.42), available - reserve))
    # 12 logical / 6 physical cores on the measured workstation resolves to 3:
    # enough throughput without saturating all SMT siblings or interactive use.
    worker_cap = max(1, min(4, physical // 2))
    disk = shutil.disk_usage(repo_root)
    peaks = []
    for manifest_path in (repo_root / "data" / "field").glob("*/cells/tf_0001/manifest.json"):
        try:
            import json
            peak = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "process_peak_working_set_bytes")
            if peak:
                peaks.append(int(peak))
        except (OSError, ValueError):
            pass
    measured_tf1_peak = max(peaks) if peaks else None
    return ResourcePolicy(logical, physical, total, available, memory_load, reserve,
                          budget, worker_cap, disk.free, 2 * GIB, measured_tf1_peak)


def estimated_cell_peak_bytes(tf_minutes: int, measured_tf1_peak_bytes: int | None = None) -> int:
    """Conservative fit to the measured indexed TF1 and lighter cells."""
    peak1 = measured_tf1_peak_bytes or int(1.8 * GIB)
    fixed = 300 * 1024 ** 2
    return int(fixed + max(fixed, peak1 - fixed) / math.sqrt(tf_minutes))


def process_peak_working_set_bytes() -> int:
    """Peak resident working set for the current cell-building process."""
    if os.name == "nt":
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current = ctypes.windll.kernel32.GetCurrentProcess
        get_current.restype = wintypes.HANDLE
        handle = get_current()
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters),
                               wintypes.DWORD]
        get_memory.restype = wintypes.BOOL
        if not get_memory(
                handle, ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak_kib * 1024)
