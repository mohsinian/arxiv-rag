"""Process-wide logging setup, driven by the LOG_LEVEL setting (API and CLI)."""

import logging
import warnings

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# These libraries log every HTTP request / model load at INFO, which buries
# the one line a CLI user actually wants. They stay quiet even at
# LOG_LEVEL=DEBUG; raise them explicitly if you ever need that firehose.
# huggingface_hub is pinned at ERROR specifically to drop the recurring
# "set HF_TOKEN" nag when serving from a warm cache.
_QUIET_LOGGERS = {
    "httpx": logging.WARNING,
    "httpx2": logging.WARNING,  # the openai SDK vendors httpx under this name
    "sentence_transformers": logging.WARNING,
    "huggingface_hub": logging.ERROR,
}


def enforce_quiet_loggers() -> None:
    """(Re-)apply the third-party quiet levels.

    sentence-transformers and huggingface_hub reconfigure their loggers while
    loading a model, so this must run again after the load, not just at startup.
    """
    for name, quiet_level in _QUIET_LOGGERS.items():
        logging.getLogger(name).setLevel(quiet_level)
    warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")


def configure_logging(level: str) -> None:
    """Send application logs (`arxiv_rag.*`) to stderr at `level`.

    Third-party request/load chatter is pinned quiet regardless of level; a
    stderr handler is added only when nothing else (a test runner, a hosting
    framework) has configured the root logger. An unknown level falls back to
    INFO with a warning rather than crashing startup over a typo. Uvicorn's
    own loggers are left as uvicorn sets them.
    """
    resolved = logging.getLevelName(level.strip().upper())
    valid = isinstance(resolved, int)
    root = logging.getLogger()
    root.setLevel(resolved if valid else logging.INFO)
    logging.getLogger("arxiv_rag").setLevel(resolved if valid else logging.INFO)
    enforce_quiet_loggers()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
    if not valid:
        logging.getLogger(__name__).warning("unknown LOG_LEVEL %r; using INFO", level)
