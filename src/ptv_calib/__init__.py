# Disable OpenCV TIFF (and other) warnings when the package is used
import logging
import os
import sys
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")  # or "OFF" on some builds

import cv2
try:
    # OpenCV 4.x: 0 = SILENT
    cv2.utils.logging.setLogLevel(0)
except AttributeError:
    pass

# ANSI color codes for terminal output
_LOG_COLORS = {
    logging.DEBUG: "\033[36m",    # cyan
    logging.INFO: "\033[32m",     # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",    # red
    logging.CRITICAL: "\033[35m", # magenta
}
_RESET = "\033[0m"


class _ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = _LOG_COLORS.get(record.levelno, _RESET)
        record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(record)


# Show calibration progress by default if no logging is configured
_ptv_logger = logging.getLogger("ptv_calib")
if not _ptv_logger.handlers and not logging.root.handlers:
    _handler = logging.StreamHandler()
    _use_color = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    if _use_color:
        _handler.setFormatter(_ColoredFormatter("%(levelname)s: %(message)s"))
    else:
        _handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    _ptv_logger.addHandler(_handler)
    _ptv_logger.setLevel(logging.INFO)
