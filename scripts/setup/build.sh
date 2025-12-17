#!/usr/bin/env bash
set -euo pipefail

bash scripts/setup/generate_env.sh
bash scripts/setup/prepare_3dgrut.sh
# build the docker images
docker compose --env-file .env -f docker/compose/docker-compose.yml build