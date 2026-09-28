import pytest

from arxiv_rag.rag.embeddings import (
    EmbeddingConfigError,
    EmbeddingProvider,
    check_embedding_dimension,
    passage_text,
)

pytestmark = pytest.mark.unit


def test_passage_text_combines_title_and_abstract():
    assert passage_text("A Title", "An abstract.") == "A Title\n\nAn abstract."


def test_passage_text_strips_empty_parts():
    assert passage_text("A Title", "  ") == "A Title"


def test_dimension_mismatch_raises_with_actionable_message():
    with pytest.raises(EmbeddingConfigError, match=r"EMBED_DIM.*reindex"):
        check_embedding_dimension(model_name="some/model", actual=384, expected=768)


def test_matching_dimension_passes():
    check_embedding_dimension(model_name="some/model", actual=768, expected=768)


def test_protocol_shape_is_satisfied_by_a_minimal_fake():
    class FakeEmbedder:
        model_name = "fake"
        dimension = 768

        def embed_texts(self, texts):
            return [[0.0] * 768 for _ in texts]

        def embed_query(self, query):
            return [0.0] * 768

    assert isinstance(FakeEmbedder(), EmbeddingProvider)
