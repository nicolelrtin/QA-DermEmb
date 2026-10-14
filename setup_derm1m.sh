#!/usr/bin/env bash
# Fetch the Derm1M repo, whose open_clip fork is required to load DermLIP-PanDerm
# (stock open_clip_torch cannot build the PanDerm CAE vision tower).
# Pinned to a known commit so results are reproducible. Safe to re-run.
set -euo pipefail

DERM1M_COMMIT="92d8fa9d9775aaf65bf3150c998d32daf4f1d32f"  # main @ 2026-08-11
ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="$ROOT/third_party/Derm1M"

if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$ROOT/third_party"
  git clone https://github.com/SiyuanYan1/Derm1M.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$DERM1M_COMMIT"
echo "Derm1M ready at $DEST ($(git -C "$DEST" rev-parse --short HEAD))"
