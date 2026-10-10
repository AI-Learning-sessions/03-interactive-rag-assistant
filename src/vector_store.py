
import uuid

import chromadb
from langchain_core.documents import Document


class VectorStore:
    """
    ChromaDB wrapper for storing and searching document embeddings.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "chroma_db",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        """
        Store documents, metadata, and embeddings in ChromaDB.
        """

        if len(documents) != len(embeddings):
            raise ValueError(
                "The number of documents must match the number of embeddings."
            )

        if not documents:
            return

        ids = [
            str(uuid.uuid4())
            for _ in documents
        ]

        self.collection.add(
            ids=ids,
            documents=[
                document.page_content
                for document in documents
            ],
            metadatas=[
                document.metadata
                for document in documents
            ],
            embeddings=embeddings,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
    ):
        """
        Search for the most semantically similar documents.
        """

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        count = self.collection.count()

        if count == 0:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "ids": [[]],
                "distances": [[]],
            }

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
        )
