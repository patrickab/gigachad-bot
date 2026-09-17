from types import SimpleNamespace

from lib.image_paths import resolve_chat_image_paths


class AssetStoreStub:
    def __init__(self, asset_path: str, content: bytes) -> None:
        self.asset = SimpleNamespace(logical_path=asset_path, content=content)
        self.read_keys: list[str] = []

    def read(self, logical_path: str):
        self.read_keys.append(logical_path)
        return self.asset


class ImageClientStub:
    def __init__(self) -> None:
        self.downscale_input: bytes | None = None

    def downscale_img(self, image: bytes, *, max_tokens: int) -> str:
        self.downscale_input = image
        return "data:image/jpeg;base64,cmVzaXplZA=="


def test_resolve_chat_images_reads_postgres_upload_without_nextcloud_copy() -> None:
    """A Postgres upload reaches the LLM through its database artifact key."""
    assets = AssetStoreStub("attachment/chat/chat-123/diagram.png", b"image-bytes")

    resolved = resolve_chat_image_paths(
        client=object(),
        chat_id="chat-123",
        slug=None,
        filenames=["diagram.png"],
        downscale=False,
        assets=assets,
    )

    assert resolved == [b"image-bytes"]
    assert assets.read_keys == ["attachment/chat/chat-123/diagram.png"]


def test_resolve_chat_images_downscales_postgres_bytes_without_nextcloud_copy() -> None:
    assets = AssetStoreStub("attachment/chat/chat-123/diagram.png", b"image-bytes")
    client = ImageClientStub()

    resolved = resolve_chat_image_paths(
        client=client,
        chat_id="chat-123",
        slug=None,
        filenames=["diagram.png"],
        downscale=True,
        assets=assets,
    )

    assert resolved == ["data:image/jpeg;base64,cmVzaXplZA=="]
    assert client.downscale_input == b"image-bytes"
