"""Cost accounting (fit/predict wall-clock)."""

from __future__ import annotations

import time
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self.elapsed = 0.0

    @contextmanager
    def measure(self):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.elapsed = time.perf_counter() - t0
