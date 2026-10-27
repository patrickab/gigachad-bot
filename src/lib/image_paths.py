"""Resolve chat upload filenames into image payloads."""

from pathlib import PurePosixPath

from agent_sandbox import PromptImage
from llm_baseclient.client import LLMClient

from lib.asset_store import Asset, AssetStore
from lib.data_store import StorageNotFoundError
from lib.storage_namespace import chat_upload

_DEFAULT_MAX_TOKENS = 2048


def _read_upload(assets: AssetStore, chat_id: str, slug: str | None, name: str) -> Asset | None:
    """Read a chat upload asset, or None when it is missing or the name is invalid."""
    try:
        return assets.read(chat_upload(chat_id, name, slug))
    except (StorageNotFoundError, ValueError):
        return None


def resolve_sandbox_prompt_images(chat_id: str, slug: str | None, filenames: list[str], assets: AssetStore) -> tuple[PromptImage, ...]:
    """Read only this user's chat uploads for the workspace agent."""
    images: list[PromptImage] = []
    for name in filenames:
        asset = _read_upload(assets, chat_id, slug, name)
        if asset and asset.kind == "upload" and asset.mime.startswith("image/") and asset.content is not None:
            images.append(PromptImage(PurePosixPath(asset.logical_path).name, asset.content))
    return tuple(images)


def resolve_chat_image_paths(
    client: LLMClient,
    chat_id: str,
    slug: str | None,
    filenames: list[str],
    downscale: bool,
    assets: AssetStore,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> list[bytes | str]:
    """Load chat upload images, optionally downscaled to *max_tokens*. Missing files are skipped."""
    images: list[bytes | str] = []
    for name in filenames:
        asset = _read_upload(assets, chat_id, slug, name)
        if asset is None or asset.content is None:
            continue
        content = asset.content
        images.append(client.downscale_img(content, max_tokens=max_tokens) if downscale else content)
    return images
