#!/usr/bin/env bash
# Fetch model weights into weights/ and verify them (sha256). Run once, WITH
# internet, before the offline evaluation (task.pdf: "weights/download.sh is run
# once, with internet, before evaluation"). Idempotent: verified files are kept.
# Total must stay <= 5 GB.
set -euo pipefail
cd "$(dirname "$0")"

if command -v sha256sum >/dev/null 2>&1; then SHA=(sha256sum); else SHA=(shasum -a 256); fi
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi

verify() {  # file sha256 -- compare strings: `-c` differs between GNU, BSD (macOS) and shasum
  [ -f "$1" ] && [ "$("${SHA[@]}" "$1" | cut -d' ' -f1)" = "$2" ]
}

fetch_url() {  # name url sha256
  local name=$1 url=$2 sha=$3
  if ! verify "$name" "$sha"; then
    echo "downloading $name"
    rm -f "$name.part"
    curl -fL --retry 5 --retry-delay 5 --retry-all-errors -o "$name.part" "$url"
    mv "$name.part" "$name"
  fi
  verify "$name" "$sha" || { echo "ERROR: checksum mismatch for $name"; exit 1; }
  echo "ok  $name"
}

# YOLO11n, COCO-pretrained (Part A detector + Part B risk detector), 5.6 MB, AGPL-3.0.
# Also committed in the repository, so this normally only verifies it.
fetch_url yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt \
  0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1

# Qwen3-VL-2B-Instruct (VLM verifier), 4.26 GB, Apache-2.0, pinned revision.
QWEN_DIR=qwen3-vl-2b-instruct
QWEN_REV=89644892e4d85e24eaac8bacfd4f463576704203
QWEN_SHA=7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0
if ! verify "$QWEN_DIR/model.safetensors" "$QWEN_SHA"; then
  for attempt in 1 2 3; do
    echo "downloading Qwen3-VL-2B-Instruct (attempt $attempt)"
    if "$PY" - "$QWEN_REV" "$QWEN_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-VL-2B-Instruct", revision=sys.argv[1], local_dir=sys.argv[2],
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "tokenizer*", "vocab.json", "merges.txt"])
PY
    then break; fi
    sleep 10
  done
fi
verify "$QWEN_DIR/model.safetensors" "$QWEN_SHA" || { echo "ERROR: Qwen3-VL weights missing or corrupt"; exit 1; }
echo "ok  $QWEN_DIR/model.safetensors"

total_kb=$(du -sk . | cut -f1)
echo "weights total: $((total_kb / 1024)) MB (limit 5120 MB)"
[ "$total_kb" -le $((5 * 1024 * 1024)) ] || { echo "ERROR: weights exceed 5 GB"; exit 1; }
