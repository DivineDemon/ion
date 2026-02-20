#!/usr/bin/env bash
# Download and extract LRA ListOps data for run_lra.py.
# Puts files under data/lra/lra_release/listops-1000/
# If the official URL fails (often returns an error page), falls back to generating data locally.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${1:-$ROOT/data/lra}"
LRA_URL="https://storage.googleapis.com/long-range-arena/lra_release.gz"
LISTOPS_DIR="$DATA_DIR/lra_release/listops-1000"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Download if missing
if [[ ! -f lra_release.gz ]]; then
  echo "Downloading LRA release..."
  curl -L -o lra_release.gz "$LRA_URL"
fi

# Validate: real gzip files start with 1f 8b and are typically > 1MB
if [[ -f lra_release.gz ]]; then
  SIZE=$(stat -f%z lra_release.gz 2>/dev/null || stat -c%s lra_release.gz 2>/dev/null)
  MAGIC=$(head -c 2 lra_release.gz | xxd -p 2>/dev/null || true)
  if [[ "$MAGIC" != "1f8b" ]] || [[ "${SIZE:-0}" -lt 1000000 ]]; then
    echo "Downloaded file is not a valid gzip (got ${SIZE:-0} bytes). Removing and using fallback."
    rm -f lra_release.gz
  fi
fi

# Extract if we have a valid archive
if [[ -f lra_release.gz ]]; then
  if [[ ! -d lra_release ]]; then
    echo "Extracting..."
    gunzip -k lra_release.gz 2>/dev/null || true
    if [[ -f lra_release ]]; then
      tar -xf lra_release 2>/dev/null || true
    fi
  fi
  if [[ ! -d "$LISTOPS_DIR" ]]; then
    echo "Extracting listops-1000 from archive..."
    (gzip -dc lra_release.gz 2>/dev/null || cat lra_release 2>/dev/null) | tar -x 2>/dev/null || true
  fi
fi

# Fallback: generate ListOps data locally (no TensorFlow, matches LRA format)
if [[ ! -d "$LISTOPS_DIR" ]]; then
  echo "Generating ListOps data locally (official URL unavailable)..."
  mkdir -p "$LISTOPS_DIR"
  PYTHON="${PYTHON:-python3}"
  if ! command -v "$PYTHON" &>/dev/null; then
    PYTHON="python"
  fi
  "$PYTHON" "$ROOT/scripts/generate_listops.py" \
    --output_dir "$LISTOPS_DIR" \
    --num_train 96000 \
    --num_val 2000 \
    --num_test 2000 \
    --min_length 500 \
    --max_length 2000
fi

if [[ -d "$LISTOPS_DIR" ]]; then
  echo "ListOps data at: $LISTOPS_DIR"
  ls -la "$LISTOPS_DIR"
else
  echo "Failed. Manual: download from $LRA_URL, extract listops-1000 to $LISTOPS_DIR, or run: $ROOT/scripts/generate_listops.py --output_dir $LISTOPS_DIR"
  exit 1
fi
