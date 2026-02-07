#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <checkpoint_path> <output_dir> <input_dir>"
    exit 1
fi

HOST_CHECKPOINT="$(realpath "$1")"
HOST_OUTPUT="$(realpath "$2")"
HOST_INPUT="$(realpath "$3")"
mkdir -p "${HOST_OUTPUT}"

docker compose --env-file .env -f docker/compose/docker-compose.yml run --rm \
  -v "${HOST_CHECKPOINT}:/checkpoint.pt" \
  -v "${HOST_INPUT}:/input:ro" \
  -v "${HOST_OUTPUT}:/output" \
  -v /dev/shm:/dev/shm \
  3dgrut_user \
  conda run -n 3dgrut --no-capture-output \
    python render.py --checkpoint /checkpoint.pt --out-dir /output