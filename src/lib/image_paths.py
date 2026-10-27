"""Resolve chat upload filenames into image payloads."""

from collections.abc import Iterator
from pathlib import PurePosixPath

from agent_sandbox import PromptImage
from llm_baseclient.client import LLMClient

from lib.asset_store import Asset, AssetStore
from lib.data_store import StorageNotFoundError
from lib.storage_namespace import chat_upload

_DEFAULT_MAX_TOKENS = 2048


def _chat_uploads(assets: AssetStore, chat_id: str, slug: str | None, filenames: list[str]) -> Iterator[Asset]:
    """Each stored chat upload, skipping missing files and invalid names."""
    for name in filenames:
        try:
            asset = assets.read(chat_upload(chat_id, name, slug))
        except (StorageNotFoundError, ValueError):
            continue
        yield asset


def resolve_sandbox_prompt_images(chat_id: str, slug: str | None, filenames: list[str], assets: AssetStore) -> tuple[PromptImage, ...]:
    """Read only this user's chat uploads for the workspace agent."""
    return tuple(
        PromptImage(PurePosixPath(asset.logical_path).name, asset.content)
        for asset in _chat_uploads(assets, chat_id, slug, filenames)
        if asset.content is not None and asset.kind == "upload" and asset.mime.startswith("image/")
    )


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
    return [
        client.downscale_img(asset.content, max_tokens=max_tokens) if downscale else asset.content
        for asset in _chat_uploads(assets, chat_id, slug, filenames)
        if asset.content is not None
    ]
