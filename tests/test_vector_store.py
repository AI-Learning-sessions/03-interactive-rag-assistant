
from langchain_core.documents import Document

from src.vector_store import VectorStore


def test_add_documents_stores_chunks(tmp_path):
    store = VectorStore(
        collection_name="test_documents",
        persist_directory=str(tmp_path),
    )

    documents = [
        Document(
            page_content="TCP provides reliable data delivery.",
            metadata={"source": "test.pdf", "page": 1},
        ),
        Document(
            page_content="UDP is a connectionless protocol.",
            metadata={"source": "test.pdf", "page": 2},
        ),
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store.add_documents(documents, embeddings)

    assert store.collection.count() == 2


def test_search_returns_document_results(tmp_path):
    store = VectorStore(
        collection_name="search_test",
        persist_directory=str(tmp_path),
    )

    document = Document(
        page_content="TCP provides reliable delivery.",
        metadata={"source": "test.pdf", "page": 3},
    )

    store.add_documents([document], [[1.0, 0.0, 0.0]])

    results = store.search([1.0, 0.0, 0.0], top_k=1)

    assert len(results["documents"][0]) == 1
    assert results["metadatas"][0][0]["page"] == 3


def test_search_handles_empty_collection(tmp_path):
    store = VectorStore(
        collection_name="empty_test",
        persist_directory=str(tmp_path),
    )

    results = store.search([1.0, 0.0, 0.0])

    assert results["documents"] == [[]]
    assert results["metadatas"] == [[]]


def test_add_documents_rejects_mismatched_embeddings(tmp_path):
    store = VectorStore(
        collection_name="validation_test",
        persist_directory=str(tmp_path),
    )

    documents = [
        Document(page_content="Test document", metadata={"page": 1})
    ]

    try:
        store.add_documents(documents, [])
    except ValueError as error:
        assert "number of documents" in str(error)
    else:
        raise AssertionError("Expected ValueError")
