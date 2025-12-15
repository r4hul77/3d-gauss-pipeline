#!/usr/bin/env bash
set -euo pipefail

VIDEO_PATH=$1        # e.g. /input/my_video.mkv
FRAME_STRIDE=$2
OUTPUT_DIR=$3

mkdir -p "${OUTPUT_DIR}"

echo "[COLMAP] Extracting frames"
echo "  Video: ${VIDEO_PATH}"
echo "  Stride: ${FRAME_STRIDE}"
echo "  Output: ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}/ffmpeg_imgs"

ffmpeg -i "${VIDEO_PATH}" \
  -vf "select=not(mod(n\,${FRAME_STRIDE}))" \
  -vsync vfr \
  "${OUTPUT_DIR}/ffmpeg_imgs/image_%04d.jpg"
