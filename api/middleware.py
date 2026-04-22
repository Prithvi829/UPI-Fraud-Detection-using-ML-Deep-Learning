"""
api/middleware.py — Request logging and rate limiting stub.

Provides before/after request hooks for the Flask application factory.
"""

from __future__ import annotations

import logging
import time

from flask import request, g

from config import LOG_DATE_FORMAT, LOG_FORMAT

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def set_start_time() -> None:
    """Record request start time for latency logging."""
    g.start_time = time.time()


def log_request(response):
    """Log each request with method, path, status, and elapsed time.

    Args:
        response: Flask response object.

    Returns:
        Unmodified response object.
    """
    elapsed = round((time.time() - getattr(g, "start_time", time.time())) * 1000, 2)
    logger.info(
        "UPI API | %s %s | status=%s | %.1fms",
        request.method,
        request.path,
        response.status_code,
        elapsed,
    )
    return response