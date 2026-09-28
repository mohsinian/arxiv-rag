import pytest

pytestmark = pytest.mark.unit


def test_package_imports():
    import arxiv_rag

    assert isinstance(arxiv_rag.__version__, str)
    assert arxiv_rag.__version__
