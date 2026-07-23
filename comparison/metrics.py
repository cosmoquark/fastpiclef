"""Timing helper: median wall time over repetitions, after a warm-up call."""

import statistics
import time


def timed(fn, reps=5):
    """Return (last_value, median_wall_seconds). One warm-up run is discarded."""
    value = fn()  # warm-up (JIT/allocation/first-touch)
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        value = fn()
        samples.append(time.perf_counter() - t0)
    return value, statistics.median(samples)
