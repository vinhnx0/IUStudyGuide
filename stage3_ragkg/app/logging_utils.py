# === stage3_ragkg/app/logging_utils.py ===
from __future__ import annotations

import functools
import logging
import sys
import time
from typing import Any, Callable, Dict, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def setup_logging(cfg: Optional[Dict[str, Any]] = None) -> None:
    """
    Configure the root logger based on an optional config dict.

    Expected structure (under cfg["logging"]):

        logging:
          level: "INFO"        # DEBUG, INFO, WARNING, ERROR, CRITICAL
          to_file: false
          filename: "stage3.log"

    If cfg or cfg["logging"] is missing, falls back to:
        level = INFO
        to_file = False

    The function is idempotent with respect to handlers:
    if handlers already exist on the root logger, it will not add new ones.
    """
    logging_cfg: Dict[str, Any] = {}
    if isinstance(cfg, dict):
        logging_cfg = cfg.get("logging", {}) or {}

    level_name = str(logging_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    to_file = bool(logging_cfg.get("to_file", False))
    filename = str(logging_cfg.get("filename", "stage3.log"))

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid re-adding handlers if already configured
    if root_logger.handlers:
        return

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

    # Stream handler to stdout
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # Optional file handler
    if to_file:
        file_handler = logging.FileHandler(filename, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Helper to obtain a logger.

    If name is None, returns a logger for this module.
    Otherwise returns logging.getLogger(name).
    """
    return logging.getLogger(name or __name__)


def log_call(level: int = logging.DEBUG, include_result: bool = False) -> Callable[[F], F]:
    """
    Decorator to log function entry, arguments, execution time, and optionally result.

    Usage example:

        @log_call(level=logging.INFO, include_result=False)
        def decide_route(...):
            ...

    The logger name is "<module>.<qualname>" so you can filter per-function.
    """
    def decorator(func: F) -> F:
        logger = logging.getLogger(f"{func.__module__}.{func.__qualname__}")

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            # Avoid heavy repr on large objects by truncating args/kwargs in logs
            try:
                logger.log(level, "CALL %s args=%r kwargs=%r", func.__name__, args, kwargs)
            except Exception:
                logger.log(level, "CALL %s (args/kwargs repr failed)", func.__name__)

            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                if include_result:
                    try:
                        logger.log(
                            level,
                            "RETURN %s in %.3fs result=%r",
                            func.__name__,
                            duration,
                            result,
                        )
                    except Exception:
                        logger.log(
                            level,
                            "RETURN %s in %.3fs (result repr failed)",
                            func.__name__,
                            duration,
                        )
                else:
                    logger.log(level, "RETURN %s in %.3fs", func.__name__, duration)
                return result
            except Exception:
                duration = time.perf_counter() - start
                logger.exception("ERROR in %s after %.3fs", func.__name__, duration)
                raise

        return wrapper  # type: ignore[return-value]

    return decorator
