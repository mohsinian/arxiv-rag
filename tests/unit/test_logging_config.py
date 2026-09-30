import logging

import pytest

from arxiv_rag.logging_config import _QUIET_LOGGERS, configure_logging

pytestmark = pytest.mark.unit


@pytest.fixture
def clean_loggers():
    root = logging.getLogger()
    saved = {
        "root": (root.handlers[:], root.level),
        "arxiv_rag": logging.getLogger("arxiv_rag").level,
    }
    for name in _QUIET_LOGGERS:
        saved[name] = logging.getLogger(name).level
    root.handlers = []
    yield root
    root.handlers, root.level = saved["root"]
    logging.getLogger("arxiv_rag").setLevel(saved["arxiv_rag"])
    for name, level in saved.items():
        if name not in ("root", "arxiv_rag"):
            logging.getLogger(name).setLevel(level)


def test_log_level_setting_drives_the_root_logger(clean_loggers):
    configure_logging("debug")
    assert clean_loggers.level == logging.DEBUG
    assert logging.getLogger("arxiv_rag").level == logging.DEBUG


def test_unknown_log_level_falls_back_to_info(clean_loggers):
    configure_logging("chatty")
    assert clean_loggers.level == logging.INFO
    assert logging.getLogger("arxiv_rag").level == logging.INFO


def test_request_chatter_stays_quiet_even_at_debug(clean_loggers):
    configure_logging("debug")
    for name in ("httpx", "httpx2", "sentence_transformers"):
        assert logging.getLogger(name).level == logging.WARNING
    assert logging.getLogger("huggingface_hub").level == logging.ERROR
