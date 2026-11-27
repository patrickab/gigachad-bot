from agent_sandbox import PromptImage

from lib.image_paths import resolve_sandbox_prompt_images
from tests.fakes import MemoryAssetStore


def _assets_with_upload(mime: str | None = None) -> MemoryAssetStore:
    assets = MemoryAssetStore()
    assets.write("upload", "attachment/chat/chat-123/diagram.png", b"image-bytes", mime=mime)
    return assets


def test_resolve_sandbox_images_forwards_only_owned_image_uploads() -> None:
    assets = _assets_with_upload("image/png")

    resolved = resolve_sandbox_prompt_images("chat-123", None, ["diagram.png", "../host-path.png"], assets)

    assert isinstance(resolved[0], PromptImage)
    assert [(image.filename, image.content) for image in resolved] == [("diagram.png", b"image-bytes")]
    assert assets.read_keys == ["attachment/chat/chat-123/diagram.png"]
