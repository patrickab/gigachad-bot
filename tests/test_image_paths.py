from agent_sandbox import PromptImage

from lib.image_paths import resolve_chat_image_paths, resolve_sandbox_prompt_images
from tests.fakes import MemoryAssetStore


def _assets_with_upload(mime: str | None = None) -> MemoryAssetStore:
    assets = MemoryAssetStore()
    assets.write("upload", "attachment/chat/chat-123/diagram.png", b"image-bytes", mime=mime)
    return assets


def test_resolve_chat_images_reads_postgres_upload_without_nextcloud_copy() -> None:
    """A Postgres upload reaches the LLM through its database artifact key."""
    assets = _assets_with_upload()

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


def test_resolve_sandbox_images_forwards_only_owned_image_uploads() -> None:
    assets = _assets_with_upload("image/png")

    resolved = resolve_sandbox_prompt_images("chat-123", None, ["diagram.png", "../host-path.png"], assets)

    assert isinstance(resolved[0], PromptImage)
    assert [(image.filename, image.content) for image in resolved] == [("diagram.png", b"image-bytes")]
    assert assets.read_keys == ["attachment/chat/chat-123/diagram.png"]
