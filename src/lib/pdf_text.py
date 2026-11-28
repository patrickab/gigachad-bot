"""The text layer a PDF carries itself, offered as a fast alternative to a MinerU extraction."""

import logging

import pypdfium2 as pdfium

log = logging.getLogger(__name__)


def text_layer(content: bytes) -> str | None:
    """The embedded text of the PDF *content*, pages separated by a blank line, or None when it has none (a scan) or does not open."""
    try:
        pdf = pdfium.PdfDocument(content)
    except pdfium.PdfiumError:
        log.warning("Could not open PDF to read its text layer")
        return None
    try:
        pages = []
        for page in pdf:
            textpage = page.get_textpage()
            pages.append(textpage.get_text_bounded().replace("\r\n", "\n").strip())
            textpage.close()
            page.close()
    finally:
        pdf.close()
    text = "\n\n".join(page for page in pages if page)
    return text or None
