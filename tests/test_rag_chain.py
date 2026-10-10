
from unittest.mock import Mock

from src.rag_chain import RAGChain


def test_rag_chain_returns_answer_and_sources():
    retriever = Mock()
    llm = Mock()

    retriever.retrieve.return_value = {
        "documents": [
            ["TCP provides reliable data delivery."]
        ],
        "metadatas": [
            [{"source": "network_protocols.pdf", "page": 8}]
        ],
    }

    llm.generate.return_value = (
        "TCP provides reliable, ordered data delivery."
    )

    rag = RAGChain(retriever, llm)

    result = rag.answer("What does TCP do?")

    assert result["answer"] == (
        "TCP provides reliable, ordered data delivery."
    )
    assert result["sources"] == [
        {"source": "network_protocols.pdf", "page": 8}
    ]

    retriever.retrieve.assert_called_once_with("What does TCP do?")
    llm.generate.assert_called_once()


def test_rag_chain_returns_document_not_found_fallback():
    retriever = Mock()
    llm = Mock()

    retriever.retrieve.return_value = {
        "documents": [
            ["An unrelated networking passage."]
        ],
        "metadatas": [
            [{"source": "network_protocols.pdf", "page": 1}]
        ],
    }

    llm.generate.return_value = (
        "I could not find the answer in the provided document."
    )

    rag = RAGChain(retriever, llm)

    result = rag.answer("What was the first human settlement on Mars?")

    assert result["answer"] == (
        "I could not find the answer in the provided document."
    )
    assert len(result["sources"]) == 1
