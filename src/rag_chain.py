from src.retriever import Retriever
from src.llm import GeminiLLM


class RAGChain:
    """
    Combines document retrieval with Gemini generation.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: GeminiLLM,
    ):
        self.retriever = retriever
        self.llm = llm

    def answer(self, question: str) -> dict:
        """
        Retrieve relevant documents and generate
        an answer based on those documents.
        """

        results = self.retriever.retrieve(question)

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        context_parts = []

        for document, metadata in zip(documents, metadatas):
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "Unknown")

            context_parts.append(
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content:\n{document}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""
You are a helpful document-based question answering assistant.

Answer the user's question using ONLY the provided document context.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided document."

Do not make up information.

Keep the answer clear and concise.

User question:
{question}

Document context:
{context}
"""

        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": metadatas,
        }