from lib.attachments import ChatUploads
from tests.fakes import MemoryAssetStore


def test_uploaded_images_are_numbered_img_nn_whatever_their_name() -> None:
    uploads = ChatUploads(MemoryAssetStore(), "chat-1")  # type: ignore[arg-type]

    names = [
        uploads.store(filename, b"px", mime, overwrite=False).name
        for filename, mime in (("image.png", "image/png"), ("image.png", "image/png"), ("photo.JPG", "image/jpeg"))
    ]

    assert names == ["img-01.png", "img-02.png", "img-03.jpg"]


def test_overwritten_image_and_deduplicated_text_keep_their_names() -> None:
    uploads = ChatUploads(MemoryAssetStore(), "chat-1")  # type: ignore[arg-type]

    assert uploads.store("plan.jpg", b"px", "image/jpeg", overwrite=True).name == "plan.jpg"
    assert [uploads.store("notes.txt", b"a", "text/plain", overwrite=False).name for _ in range(2)] == ["notes.txt", "notes (1).txt"]


def _pdf(text: str) -> bytes:
    """A one-page PDF showing *text* in Helvetica, no text layer when *text* is empty."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offsets)
    return out + b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)


def test_uploaded_pdf_returns_its_text_layer_only_when_it_has_one() -> None:
    uploads = ChatUploads(MemoryAssetStore(), "chat-1")  # type: ignore[arg-type]

    assert uploads.store("paper.pdf", _pdf("Hello world"), "application/pdf", overwrite=False).textLayer == "Hello world"
    assert uploads.store("scan.pdf", _pdf(""), "application/pdf", overwrite=False).textLayer is None
    assert uploads.store("broken.pdf", b"not a pdf", "application/pdf", overwrite=False).textLayer is None
