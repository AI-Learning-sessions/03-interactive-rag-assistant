
import hashlib
import os
import re
import tempfile

import streamlit as st

from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.ingestion import DocumentIngestion
from src.retriever import Retriever
from src.llm import GeminiLLM
from src.rag_chain import RAGChain


st.set_page_config(
    page_title="Interactive RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Interactive RAG Assistant")
st.write("Upload a PDF and ask questions about its contents.")


@st.cache_resource
def get_embedding_model():
    return EmbeddingModel()


@st.cache_resource
def get_llm():
    return GeminiLLM()


def build_rag(collection_name: str, persist_directory: str):
    embedding_model = get_embedding_model()

    vector_store = VectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    retriever = Retriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
        top_k=4,
    )

    return RAGChain(
        retriever=retriever,
        llm=get_llm(),
    )


def safe_filename(filename: str) -> str:
    name = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)[:40] or "document"


uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
)

if uploaded_file is not None:
    st.write(f"**Selected file:** {uploaded_file.name}")

    file_bytes = uploaded_file.getvalue()

    if st.button("Process Document", type="primary"):
        if not file_bytes:
            st.error("The uploaded file is empty.")
        else:
            content_hash = hashlib.sha256(file_bytes).hexdigest()
            collection_name = f"doc_{content_hash[:32]}"

            os.makedirs("chroma_db", exist_ok=True)

            try:
                with tempfile.TemporaryDirectory() as temporary_directory:
                    temporary_path = os.path.join(
                        temporary_directory,
                        os.path.basename(uploaded_file.name),
                    )

                    with open(temporary_path, "wb") as temporary_file:
                        temporary_file.write(file_bytes)

                    with st.spinner("Processing your PDF..."):
                        vector_store = VectorStore(
                            collection_name=collection_name,
                            persist_directory="chroma_db",
                        )

                        existing_count = vector_store.collection.count()

                        if existing_count == 0:
                            embedding_model = get_embedding_model()

                            ingestion = DocumentIngestion(
                                embedding_model,
                                vector_store,
                            )

                            chunk_count = ingestion.process(temporary_path)
                        else:
                            chunk_count = existing_count

                st.session_state["processed_document"] = {
                    "name": uploaded_file.name,
                    "collection_name": collection_name,
                    "chunk_count": chunk_count,
                }

                st.session_state["messages"] = []

                st.success(
                    f"Document ready! {chunk_count} chunks available."
                )

            except Exception as error:
                st.error(f"Could not process the PDF: {error}")


document = st.session_state.get("processed_document")

if document:
    st.divider()
    st.subheader("Ask questions about your document")
    st.caption(
        f"Active document: {document['name']} "
        f"({document['chunk_count']} chunks)"
    )

    question = st.text_input(
        "Your question",
        key="question_input",
        placeholder="What is the TCP/IP model?",
    )

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                rag = build_rag(
                    document["collection_name"],
                    "chroma_db",
                )

                with st.spinner("Finding relevant information..."):
                    result = rag.answer(question.strip())

                st.subheader("Answer")
                st.write(result["answer"])

                st.subheader("Sources")
                seen_sources = set()

                for source in result["sources"]:
                    source_name = os.path.basename(
                        source.get("source", document["name"])
                    )
                    page = source.get("page", "Unknown")
                    source_key = (source_name, page)

                    if source_key not in seen_sources:
                        seen_sources.add(source_key)
                        st.write(f"📄 {source_name} — Page {page}")

            except Exception as error:
                st.error(f"Could not generate an answer: {error}")

else:
    st.info("Upload and process a PDF to begin asking questions.")
