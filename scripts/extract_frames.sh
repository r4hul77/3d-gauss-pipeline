#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <video_file> <frame_stride> <output_dir>"
  exit 1
fi

HOST_VIDEO=$(realpath "$1")
FRAME_STRIDE=$2
HOST_OUTPUT=$(realpath "$3")

if [ ! -f "${HOST_VIDEO}" ]; then
  echo "Error: video file not found: ${HOST_VIDEO}"
  exit 1
fi

VIDEO_DIR=$(dirname "${HOST_VIDEO}")
VIDEO_NAME=$(basename "${HOST_VIDEO}")

mkdir -p "${HOST_OUTPUT}"

docker compose -f docker/compose/docker-compose.yml run --rm \
  -u $(id -u):$(id -g) \
  -v "${VIDEO_DIR}:/input:ro" \
  -v "${HOST_OUTPUT}:/output" \
  colmap_user \
  /usr/local/bin/extract_frames.sh \
  "/input/${VIDEO_NAME}" \
  "${FRAME_STRIDE}" \
  /output