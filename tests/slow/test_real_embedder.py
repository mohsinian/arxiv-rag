import pytest

from arxiv_rag.rag.embeddings import SentenceTransformerEmbedder

pytestmark = pytest.mark.slow


def test_real_model_embeds_and_reports_dimension():
    embedder = SentenceTransformerEmbedder(
        "Alibaba-NLP/gte-modernbert-base", expected_dim=768, batch_size=4
    )
    assert embedder.model_name == "Alibaba-NLP/gte-modernbert-base"
    assert embedder.dimension == 768

    vectors = embedder.embed_texts(["hello world", "graph neural networks"])
    assert len(vectors) == 2
    assert all(len(v) == 768 for v in vectors)

    query = embedder.embed_query("what is retrieval augmented generation?")
    assert len(query) == 768
