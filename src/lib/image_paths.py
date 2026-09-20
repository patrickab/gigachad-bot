from pathlib import PurePosixPath
import re

from agent_sandbox import PromptImage
from llm_baseclient.client import LLMClient

from lib.asset_store import AssetStore
from lib.data_store import StorageNotFoundError
from lib.storage_namespace import chat_upload

_DEFAULT_MAX_TOKENS = 2048
_DATA_URI_RE = re.compile(r"data:image/\w+;base64,(.+)")


def resolve_sandbox_prompt_images(chat_id: str, slug: str | None, filenames: list[str], assets: AssetStore) -> tuple[PromptImage, ...]:
    """Read only this user's chat uploads for the workspace agent."""
    images: list[PromptImage] = []
    for name in filenames:
        try:
            asset = assets.read(chat_upload(chat_id, name, slug))
        except (StorageNotFoundError, ValueError):
            continue
        if asset.kind == "upload" and asset.mime.startswith("image/") and asset.content is not None:
            images.append(PromptImage(PurePosixPath(asset.logical_path).name, asset.content))
    return tuple(images)


def _read_image_asset(assets: AssetStore, chat_id: str, slug: str | None, name: str) -> bytes | None:
    """Read a Postgres-backed chat image without copying it into Nextcloud."""
    try:
        asset = assets.read(chat_upload(chat_id, name, slug))
    except (StorageNotFoundError, ValueError):
        return None
    return asset.content


def resolve_chat_image_paths(
    client: LLMClient,
    chat_id: str,
    slug: str | None,
    filenames: list[str],
    downscale: bool,
    assets: AssetStore,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> list[bytes | str]:
    if not filenames:
        return []
    images: list[bytes | str] = []
    for name in filenames:
        content = _read_image_asset(assets, chat_id, slug, name)
        if content is None:
            continue
        images.append(client.downscale_img(content, max_tokens=max_tokens) if downscale else content)
    return images
