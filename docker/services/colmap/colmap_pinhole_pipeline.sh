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

INPUT_DIR="$(realpath "$1")"
OUTPUT_DIR="$(realpath "$2")"

if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "[ERROR] Input directory not found: ${INPUT_DIR}"
    exit 1
fi

#############################################
# CONFIG
#############################################

OVERLAP=10
USE_GPU=1

WORKDIR="$(mktemp -d)"
STAGE1="${OUTPUT_DIR}/stage1"
STAGE2="${OUTPUT_DIR}/stage2"

mkdir -p "${STAGE1}"
mkdir -p "${STAGE2}"

#############################################
# SAFETY
#############################################

command -v colmap >/dev/null || {
    echo "[ERROR] colmap not found in PATH"
    exit 1
}

#############################################
# -------- STAGE 1: OPENCV (distorted) ------
#############################################

echo "[INFO] Stage 1: Distorted OPENCV reconstruction"

mkdir -p "${STAGE1}/sparse"

colmap feature_extractor \
    --database_path "${STAGE1}/database.db" \
    --image_path "${INPUT_DIR}" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \

colmap sequential_matcher \
    --database_path "${STAGE1}/database.db" \
    --SequentialMatching.overlap ${OVERLAP} 

colmap mapper \
    --database_path "${STAGE1}/database.db" \
    --image_path "${INPUT_DIR}" \
    --output_path "${STAGE1}/sparse"


#############################################
# -------- STAGE 1A: Chose the Best Model ------
#############################################
#############################################
# SELECT BEST SPARSE MODEL (MAX IMAGES)
#############################################

echo "[INFO] Selecting best sparse model (max registered images)"

BEST_MODEL=""
MAX_IMAGES=0

for MODEL_DIR in "${STAGE1}/sparse/"*; do
    [[ -d "${MODEL_DIR}" ]] || continue

    MODEL_ID="$(basename "${MODEL_DIR}")"

    TMP_TXT="$(mktemp -d)"
    colmap model_converter \
        --input_path "${MODEL_DIR}" \
        --output_path "${TMP_TXT}" \
        --output_type TXT >/dev/null 2>&1

    IMAGE_COUNT=$(grep -v '^#' "${TMP_TXT}/images.txt" | wc -l)
    rm -rf "${TMP_TXT}"

    echo "  Model ${MODEL_ID}: ${IMAGE_COUNT} images"

    if (( IMAGE_COUNT > MAX_IMAGES )); then
        MAX_IMAGES=${IMAGE_COUNT}
        BEST_MODEL="${MODEL_DIR}"
    fi
done

if [[ -z "${BEST_MODEL}" ]]; then
    echo "[ERROR] No valid sparse models found"
    exit 1
fi

echo "[INFO] Selected model: ${BEST_MODEL} (${MAX_IMAGES} images)"


#############################################
# -------- STAGE 2: UNDISTORT ---------------
#############################################

echo "[INFO] Stage 2: Undistorting images"

mkdir -p "${STAGE2}"

colmap image_undistorter \
    --image_path "${INPUT_DIR}" \
    --input_path "${BEST_MODEL}" \
    --output_path "${STAGE2}" \
    --output_type COLMAP

#############################################
# -------- STAGE 3: PINHOLE CLEAN REBUILD ---
#############################################

echo "[INFO] Stage 3: Clean PINHOLE reconstruction"

mkdir -p "${OUTPUT_DIR}"

colmap feature_extractor \
    --database_path "${OUTPUT_DIR}/database.db" \
    --image_path "${STAGE2}/images" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model PINHOLE \

colmap sequential_matcher \
    --database_path "${OUTPUT_DIR}/database.db" \
    --SequentialMatching.overlap ${OVERLAP} \
    --SequentialMatching.loop_detection 1 

colmap mapper \
    --database_path "${OUTPUT_DIR}/database.db" \
    --image_path "${STAGE2}/images" \
    --output_path "${STAGE2}/sparse"

#############################################
# FINALIZE OUTPUT
#############################################


rm -rf "${WORKDIR}"

echo "[DONE] Final PINHOLE dataset ready at: ${STAGE2}"
