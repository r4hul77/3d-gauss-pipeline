#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <frames_dir> <output_dir>"
    exit 1
fi

HOST_DATASET="$(realpath "$1")"
HOST_OUTPUT="$(realpath "$2")"

if [[ ! -d "${HOST_DATASET}" ]]; then
    echo "[ERROR] Dataset directory not found: ${HOST_DATASET}"
    exit 1
fi

mkdir -p "${HOST_OUTPUT}"

docker compose -f docker/compose/docker-compose.yml run --rm \
  -v "${HOST_DATASET}:/input:ro" \
  -v "${HOST_OUTPUT}:/output" \
  -v /dev/shm:/dev/shm \
  3dgrut_user \
  conda run -n 3dgrut --no-capture-output \
    python train.py --config-name apps/colmap_3dgrt.yaml \
    path=/input out_dir=/output experiment_name=3dgrt \
    export_ply.enabled=true export_usdz.enabled=true \
    export_ply.path=/output/export_last.ply export_usdz.path=/output/export_last.usdz \
    checkpoint.iterations=[20,40,60,80,100] \
    n_iterations=100