from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


class Retriever:
    """
    Handles semantic retrieval from the vector store.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        top_k: int = 4,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k

    def retrieve(self, query: str):
        """
        Retrieve the most relevant document chunks for a query.
        """

        query_embedding = self.embedding_model.embed_query(query)

        return self.vector_store.search(
            query_embedding,
            top_k=self.top_k,
        )