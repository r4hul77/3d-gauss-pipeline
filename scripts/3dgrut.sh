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

docker compose --env-file .env -f docker/compose/docker-compose.yml run --rm \
  -v "${HOST_DATASET}:/input:ro" \
  -v "${HOST_OUTPUT}:/output" \
  -v /dev/shm:/dev/shm \
  3dgrut_user \
  conda run -n 3dgrut --no-capture-output \
    python train.py --config-name apps/colmap_3dgrt.yaml \
    initialization=random initialization.num_gaussians=15625 initialization.xyz_max=2.5 initialization.xyz_min=-2.5 \
    path=/input out_dir=/output experiment_name=3dgrt \
    export_ply.enabled=true export_usdz.enabled=true \
    export_ply.path=/output/export_last.ply export_usdz.path=/output/export_last.usdz \
    checkpoint.iterations=[2000,4000,6000,8000,10000,15000,20000] \
    n_iterations=20000