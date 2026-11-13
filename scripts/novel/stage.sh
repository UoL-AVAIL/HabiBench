#!/usr/bin/env bash
# Stage immutable artifacts on a network-enabled Linux host with shared $WORK.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd -P)
cd "$repo"
stage=${HABIBENCH_STAGE:-${WORK:?Set WORK to the shared project storage}/habibench}
mkdir -p "$stage/images" "$stage/models"
stage=$(cd "$stage" && pwd -P)
export ENROOT_CACHE_PATH=${ENROOT_CACHE_PATH:-$stage/enroot-cache}
export ENROOT_TEMP_PATH=${ENROOT_TEMP_PATH:-$stage/enroot-tmp}
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_TEMP_PATH"
cd "$stage/images"
exec 9>stage.lock
flock 9
oci_sha=6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9
image="vllm-openai-v0.32.0-${oci_sha}.sqfs"
if [[ -f "$image" && -f "$image.sha256" ]]; then
    sha256sum --check "$image.sha256"
else
    temp_dir=$(mktemp -d "$stage/images/.import.XXXXXXXX")
    trap 'rm -rf -- "$temp_dir"' EXIT
    # Enroot uses @ for registry usernames. Pass the digest as the manifest reference.
    enroot import --output "$temp_dir/$image" "docker://registry-1.docker.io#vllm/vllm-openai:sha256:$oci_sha"
    (cd "$temp_dir" && sha256sum "$image" > "$image.sha256")
    mv "$temp_dir/$image" "$image"
    mv "$temp_dir/$image.sha256" "$image.sha256"
fi
cd "$repo"
python -m habibench.lucas fetch-model --model-dir "$stage/models"
printf 'Staged files in %s\n' "$stage"
