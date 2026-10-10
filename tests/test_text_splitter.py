
from langchain_core.documents import Document

from src.text_splitter import split_documents


def test_split_documents_creates_chunks():
    document = Document(
        page_content="Network protocols describe " * 100,
        metadata={"source": "test.pdf", "page": 1},
    )

    chunks = split_documents(
        [document],
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)


def test_split_documents_preserves_metadata():
    document = Document(
        page_content="TCP and UDP are transport protocols. " * 20,
        metadata={"source": "network.pdf", "page": 5},
    )

    chunks = split_documents([document])

    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.metadata["source"] == "network.pdf"
        assert chunk.metadata["page"] == 5


def test_empty_document_list_returns_no_chunks():
    assert split_documents([]) == []
