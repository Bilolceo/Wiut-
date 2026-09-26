#!/usr/bin/env bash
# Download model weights into weights/ and verify sha256 (run once, WITH internet;
# the eval machine itself never downloads anything). Total must stay <= 5 GB.
set -euo pipefail
cd "$(dirname "$0")"

fetch() {  # name url sha256
  local name=$1 url=$2 sha=$3
  if [ ! -f "$name" ] || ! echo "$sha  $name" | shasum -a 256 -c --status 2>/dev/null; then
    echo "downloading $name"
    curl -fL --retry 3 -o "$name.part" "$url"
    mv "$name.part" "$name"
  fi
  echo "$sha  $name" | shasum -a 256 -c
}

# YOLO11n, COCO-pretrained (Part A detector + Part B risk detector), 5.6 MB, AGPL-3.0
fetch yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt \
  0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1

# Qwen3-VL-2B-Instruct (VLM verifier), ~4.27 GB, Apache-2.0. Pinned revision; resumable.
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-VL-2B-Instruct", revision="89644892e4d85e24eaac8bacfd4f463576704203", local_dir="qwen3-vl-2b-instruct",
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "tokenizer*", "vocab.json", "merges.txt"])
PY

total_kb=$(du -sk . | cut -f1)
echo "weights total: $((total_kb / 1024)) MB (limit 5120 MB)"
[ "$total_kb" -le $((5 * 1024 * 1024)) ] || { echo "ERROR: weights exceed 5 GB"; exit 1; }
