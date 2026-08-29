#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
frontend_dir="$root_dir/src/frontend"
tauri_dir="$frontend_dir/src-tauri"
binary_dir="$tauri_dir/binaries"
build_dir="$root_dir/.build/pyinstaller"
target_triple="$(rustc --print host-tuple)"

if [[ "$target_triple" != *-linux-* ]]; then
  echo "This routine currently builds Linux sidecars only (got $target_triple)." >&2
  exit 1
fi

# mksquashfs: linuxdeploy's appimage plugin shells out to it to build the
# .AppImage.
for bin_pkg in mksquashfs:squashfs-tools; do
  bin="${bin_pkg%%:*}"
  pkg="${bin_pkg##*:}"
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "$bin not found; installing $pkg..." >&2
    if command -v pacman >/dev/null 2>&1; then
      sudo pacman -S --needed --noconfirm "$pkg"
    elif command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update && sudo apt-get install -y "$pkg"
    else
      echo "Unsupported package manager: install $pkg manually." >&2
      exit 1
    fi
  fi
done

mkdir -p "$binary_dir"
mkdir -p "$build_dir"
rm -rf "$binary_dir/gigachad-sidecar"

# Tauri's own appimage bundler doesn't clean this between runs, and reusing a
# stale AppDir across many builds has produced a corrupted .AppImage (files
# present on disk but silently dropped from the squashed image) - always
# start from a clean bundle output.
rm -rf "$tauri_dir/target/release/bundle"

# Next.js's .next directory has both served stale compiled CSS across
# `beforeBuildCommand` runs (cache not invalidating on globals.css changes)
# and, when only .next/cache was cleared, thrown PageNotFoundError on
# /_document from a stale/mismatched leftover build manifest. Wiping the
# whole directory avoids both - every desktop build starts from a clean slate.
rm -rf "$frontend_dir/.next"

# onedir (default without --onefile) keeps extraction cost low and sidesteps
# --onefile's 4GiB CArchive offset limit. No --clean: reuses the build cache.
#
# MinerU is bundled client-only (mineru.cli.api_client, pulled in via the lazy
# import in backend/routes/mineru.py): its OCR *server* spawns
# `sys.executable -m mineru.cli.fast_api`, which can't work from a frozen app,
# so the desktop build points at an external server via MINERU_SERVER_URL
# instead of shipping the ~12GB torch/CUDA/vllm stack. The excludes below pin
# that stack out in case anything else's import graph reaches for it.
uv run --with pyinstaller pyinstaller \
  --noconfirm \
  --name gigachad-sidecar \
  --paths "$root_dir" \
  --paths "$root_dir/src" \
  --add-data "$root_dir/prompts:prompts" \
  --collect-all gpt_researcher \
  --collect-all langchain_community \
  --collect-all langchain_litellm \
  --collect-all litellm \
  --collect-all tiktoken_ext \
  --exclude-module torch \
  --exclude-module torchvision \
  --exclude-module vllm \
  --exclude-module nvidia \
  --exclude-module triton \
  --exclude-module xformers \
  --exclude-module cupy \
  --exclude-module cv2 \
  --exclude-module spacy \
  --exclude-module llvmlite \
  --exclude-module numba \
  --exclude-module pyarrow \
  --exclude-module onnxruntime \
  --distpath "$binary_dir" \
  --workpath "$build_dir" \
  --specpath "$build_dir" \
  "$root_dir/src/backend/sidecar.py"

# linuxdeploy is itself an AppImage whose runtime needs libfuse.so.2, which
# modern systems (fuse3-only) no longer ship; self-extraction avoids FUSE.
export APPIMAGE_EXTRACT_AND_RUN=1

# productName matches the binary name (gigachad-bot), which sidesteps the Tauri
# appimage-bundler bug where the AppDir root icon is named after productName
# while the .desktop Icon= key points at the binary name.
set +e
npm --prefix "$frontend_dir" run tauri -- build --bundles appimage
build_status=$?
set -e

if [[ $build_status -ne 0 ]]; then
  # linuxdeploy reliably aborts while scanning the PyInstaller sidecar: wheel-
  # vendored libs keep RPATHs like $ORIGIN/../pillow.libs that PyInstaller
  # flattens away (it resolves them at runtime via the bootloader's
  # LD_LIBRARY_PATH), so static resolution reports e.g. libavif as missing.
  # By then the shell/webview libs are already deployed and the sidecar is
  # self-contained, so packaging the AppDir as-is yields a working AppImage.
  appdir="$tauri_dir/target/release/bundle/appimage/gigachad-bot.AppDir"
  plugin="${XDG_CACHE_HOME:-$HOME/.cache}/tauri/linuxdeploy-plugin-appimage.AppImage"
  if [[ ! -d "$appdir" || ! -x "$plugin" ]]; then
    echo "AppImage bundling failed before the AppDir was assembled." >&2
    exit "$build_status"
  fi

  bundle_dir="$(dirname "$appdir")"
  find "$bundle_dir" -maxdepth 1 -iname "*.AppImage" -delete
  (cd "$bundle_dir" && "$plugin" --appdir "$(basename "$appdir")")

  version=$(sed -n 's/.*"version": "\([^"]*\)".*/\1/p' "$tauri_dir/tauri.conf.json" | head -n1)
  mv "$bundle_dir/gigachad-bot-x86_64.AppImage" "$bundle_dir/gigachad-bot_${version}_amd64.AppImage"
fi
