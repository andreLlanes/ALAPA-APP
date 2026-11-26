""" Optional, low-overhead timing of training stages. Set the environment variable O2_PROFILE=1 to
    enable it; it then records wall time per named section and prints a summary when the process
    exits. Disabled by default, in which case every section is a no-op, so it never affects results
    or speed. GPU work is asynchronous, so timing synchronizes the device around each section.
"""

import atexit
import os
import time
from collections import defaultdict
from contextlib import contextmanager

ENABLED = os.environ.get("O2_PROFILE") == "1"
_totals = defaultdict(float)
_counts = defaultdict(int)

def _sync():
    """ Wait for pending GPU work so a section's time is the work, not just the launch.
    """
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()

@contextmanager
def section(name: str):
    """ Time the enclosed block under `name` when profiling is on, otherwise do nothing.
    """
    if not ENABLED:
        yield
        return
    _sync()
    start = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        _totals[name] += time.perf_counter() - start
        _counts[name] += 1

def report():
    """ Print the accumulated per-section wall time, slowest first.
    """
    if not ENABLED or not _totals:
        return
    width = max(len(name) for name in _totals)
    print("\n[PROFILE] wall time by stage (s):")
    for name in sorted(_totals, key=lambda k: -_totals[k]):
        print(f"    {name:<{width}}  {_totals[name]:9.2f}  ({_counts[name]} calls)")

atexit.register(report)
