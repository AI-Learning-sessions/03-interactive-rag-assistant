from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """
    Wrapper around the Sentence Transformers embedding model.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Convert multiple documents into embedding vectors.
        """

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Convert a single query into an embedding vector.
        """

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embedding.tolist()