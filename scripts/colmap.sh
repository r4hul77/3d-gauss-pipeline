#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <frames_dir> <output_dir>"
    exit 1
fi

HOST_FRAMES="$(realpath "$1")"
HOST_OUTPUT="$(realpath "$2")"

if [[ ! -d "${HOST_FRAMES}" ]]; then
    echo "[ERROR] Frames directory not found: ${HOST_FRAMES}"
    exit 1
fi

mkdir -p "${HOST_OUTPUT}"

docker compose -f docker/compose/docker-compose.yml run --rm \
  -u $(id -u):$(id -g) \
  -v "${HOST_FRAMES}:/input:ro" \
  -v "${HOST_OUTPUT}:/output" \
  colmap_user \
  /usr/local/bin/colmap_pinhole_pipeline.sh \
  /input \
  /output
