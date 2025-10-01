# logger.py
import logging, sys, json, time
from typing import Any, Mapping
from app.config import settings

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: Mapping[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "service": settings.service_name,
            "message": record.getMessage(),
        }
        if record.__dict__.get("extra"):
            base.update(record.__dict__["extra"])  # type: ignore
        return json.dumps(base, ensure_ascii=False)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter() if settings.log_json else logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
