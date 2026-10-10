
from src.document_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


class DocumentIngestion:
    """
    Handles the complete PDF ingestion pipeline.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def process(self, file_path: str) -> int:
        """
        Load, split, embed, and store a PDF.
        Returns the number of chunks stored.
        """

        documents = load_pdf(file_path)

        if not documents:
            raise ValueError(
                "The PDF contains no extractable text."
            )

        chunks = split_documents(documents)

        if not chunks:
            raise ValueError(
                "No text chunks were created from the PDF."
            )

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        embeddings = self.embedding_model.embed_documents(
            texts
        )

        self.vector_store.add_documents(
            chunks,
            embeddings,
        )

        return len(chunks)
