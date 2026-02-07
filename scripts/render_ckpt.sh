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
  -v "${HOST_INPUT}:/input" \
  -v "${HOST_OUTPUT}:/output" \
  -v /dev/shm:/dev/shm \
  3dgrut_user \
  conda run -n 3dgrut --no-capture-output \
    python train.py --config-name apps/colmap_3dgrt.yaml \
    initialization=random \
    path=/input out_dir=/output experiment_name=3dgrt \
    export_ply.enabled=true export_usdz.enabled=true \
    export_ply.path=/output/export_last.ply export_usdz.path=/output/export_last.usdz \
    resume=/checkpoint.pt \
    with_gui=True