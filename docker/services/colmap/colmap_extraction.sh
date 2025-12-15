#!/usr/bin/env bash
set -euo pipefail

#############################################
# USAGE
#############################################
# colmap_pinhole_pipeline.sh <input_dir> <output_dir>
#############################################

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    exit 1
fi

OVERLAP=10
USE_GPU=1


INPUT_DIR="$(realpath "$1")"
OUTPUT_DIR="$(realpath "$2")"


mkdir -p "${OUTPUT_DIR}/sparse"

colmap feature_extractor \
    --database_path "${OUTPUT_DIR}/database.db" \
    --image_path "${INPUT_DIR}" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \

colmap sequential_matcher \
    --database_path "${OUTPUT_DIR}/database.db" \
    --SequentialMatching.overlap ${OVERLAP} \
    --SequentialMatching.loop_detection 1 

colmap mapper \
    --database_path "${OUTPUT_DIR}/database.db" \
    --image_path "${INPUT_DIR}" \
    --output_path "${OUTPUT_DIR}/sparse"
