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
