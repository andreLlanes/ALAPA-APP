""" Record everything a run does in a text file under artifacts/logs/, for checking and for the
    paper: every printed line, plus detail sections (tables, weights, statistics) written only to
    the file. Without an open log, detail() does nothing.
"""

import platform
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_log = None
_console = None

class _Tee:
    """ Send everything printed to the console and to the log file.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()

def start(directory, name: str) -> Path:
    """ Open a new log file and copy every printed line into it until stop().
    """
    global _log, _console
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now():%Y%m%d-%H%M%S}_{name}.txt"
    _log = open(path, "w", encoding="utf-8")
    _console = sys.stdout
    sys.stdout = _Tee(_console, _log)
    return path

def stop():
    """ Close the log and restore normal printing.
    """
    global _log
    if _log is not None:
        sys.stdout = _console
        _log.close()
        _log = None

def detail(title: str, values=None, table=None):
    """ Write a detail section to the log file only: key/value lines and/or a table.
    """
    if _log is None:
        return
    lines = [f"[{title}]"]
    for key, value in (values or {}).items():
        lines.append(f"    {key}: {value}")
    if table is not None and len(table):
        frame = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
        lines += ["    " + row for row in frame.to_string(index=False).splitlines()]
    _log.write("\n".join(lines) + "\n")
    _log.flush()

def environment() -> dict:
    """ Return the software versions behind a run, for reproducibility.
    """
    import numpy
    import torch
    import xgboost
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "pandas": pd.__version__, "torch": torch.__version__,
            "xgboost": xgboost.__version__, "cuda available": torch.cuda.is_available()}
