import os
import fitz


def load_pdf(pdf_path: str) -> list[dict]:
    """Load text from a PDF file page by page.

    Returns:
        List of dicts: {text, page, source}
    """
    document = fitz.open(pdf_path)
    pages = []
    for page_number, page in enumerate(document):
        text = page.get_text()
        if text.strip():
            pages.append({
                "text":   text,
                "page":   page_number + 1,
                "source": os.path.basename(pdf_path),
            })
    document.close()
    return pages
