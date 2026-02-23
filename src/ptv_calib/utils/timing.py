"""Timing utilities for logging step durations."""
import time
import logging
from contextlib import contextmanager
from typing import Optional


@contextmanager
def timed(
    logger: logging.Logger,
    step_name: str,
    level: int = logging.INFO,
    extra_msg: str = "",
):
    """
    Context manager that logs the elapsed time for a block of code.

    Usage:
        with timed(logger, "read_images"):
            read_images(...)
        # Logs: "read_images: 1.23s"
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        msg = f"{step_name}: {elapsed:.3f}s"
        if extra_msg:
            msg = f"{msg} ({extra_msg})"
        logger.log(level, msg)
