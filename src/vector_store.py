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

        ids = [
            f"doc-{index}"
            for index in range(len(documents))
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

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )