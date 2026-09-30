import logging

import pytest

from arxiv_rag.logging_config import configure_logging

pytestmark = pytest.mark.unit


@pytest.fixture
def clean_root_logger():
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    root.handlers = []
    yield root
    root.handlers, root.level = saved_handlers, saved_level


def test_log_level_setting_drives_the_root_logger(clean_root_logger):
    configure_logging("debug")
    assert clean_root_logger.level == logging.DEBUG


def test_unknown_log_level_falls_back_to_info(clean_root_logger):
    configure_logging("chatty")
    assert clean_root_logger.level == logging.INFO
