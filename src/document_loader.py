import pymupdf
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    """
    Load a PDF file and convert each page into a LangChain Document.
    """

    documents = []

    pdf = pymupdf.open(file_path)

    try:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text().strip()

            if not text:
                continue

            document = Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "page": page_number,
                },
            )

            documents.append(document)

    finally:
        pdf.close()

    return documents