#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 path/to/config.yaml" >&2
  exit 2
fi

config=$1
test -f "$config"

if grep -q '/path/to/' "$config"; then
  echo "ERROR: replace all placeholder paths in a private local copy first." >&2
  exit 3
fi

export TOKENIZERS_PARALLELISM=false
FORCE_TORCHRUN=1 llamafactory-cli train "$config"
