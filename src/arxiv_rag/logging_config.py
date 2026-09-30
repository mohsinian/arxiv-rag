"""Process-wide logging setup, driven by the LOG_LEVEL setting (API and CLI)."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str) -> None:
    """Send application logs to stderr at `level` (e.g. "DEBUG", "info").

    The level is always applied; a stderr handler is added only when nothing
    else (a test runner, a hosting framework) has configured the root logger.
    An unknown level falls back to INFO with a warning rather than crashing
    startup over a typo. Uvicorn's own loggers are left as uvicorn sets them.
    """
    resolved = logging.getLevelName(level.strip().upper())
    valid = isinstance(resolved, int)
    root = logging.getLogger()
    root.setLevel(resolved if valid else logging.INFO)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
    if not valid:
        logging.getLogger(__name__).warning("unknown LOG_LEVEL %r; using INFO", level)
