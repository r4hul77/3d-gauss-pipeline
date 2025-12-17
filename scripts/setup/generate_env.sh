#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=".env"

UID_VAL="$(id -u)"
GID_VAL="$(id -g)"
USERNAME_VAL="$(id -un)"
# if HUGGINGFACE_TOKEN is not set, ask the user to set it later for now set it to an empty string   

if [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
    echo "Please Set the HUGGINGFACE_TOKEN If you want to use SAM3"
    HUGGINGFACE_TOKEN="EMPTY_TOKEN"
fi


cat > "${ENV_FILE}" <<EOF
ENVIRONMENT=development
GIT_DISCOVERY_ACROSS_FILESYSTEM=1
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
UID=${UID_VAL}
GID=${GID_VAL}
USERNAME=${USERNAME_VAL}
HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
EOF

echo "Generated ${ENV_FILE} with:"
echo "  UID=${UID_VAL}"
echo "  GID=${GID_VAL}"
echo "  USERNAME=${USERNAME_VAL}"
