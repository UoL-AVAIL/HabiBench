#!/usr/bin/env bash
#SBATCH --job-name=lucas-qwen
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus=nvidia_rtx_pro_6000:2
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=lucas-qwen-%j.log

# Submit from the checkout root (Slurm spools a copy of this script).
set -euo pipefail
scope=${1:-tune}
case "$scope" in smoke|tune|audit|all) ;; *) echo 'Use smoke, tune, audit or all' >&2; exit 2;; esac
if [[ "$scope" == all || "$scope" == audit ]] && [[ -z ${3:-} ]]; then
    echo 'Supply a run name and a frozen tuning run name as arguments 2 and 3' >&2
    exit 2
fi
run_name=${2:-novel-context-$scope}
freeze_name=${3:-}
for label in "$run_name" "$freeze_name"; do
    if [[ -n "$label" && ! "$label" =~ ^[[:alnum:]][[:alnum:]._-]*$ ]]; then
        echo "Use a run name, not a path: $label" >&2; exit 2
    fi
done

repo=$(cd "${HABIBENCH_REPO:-${SLURM_SUBMIT_DIR:?}}" && pwd -P)
data=$(cd "${HABIBENCH_DATA:-$repo/data/lucas_tessera}" && pwd -P)
stage=$(cd "${HABIBENCH_STAGE:-${WORK:?}/habibench}" && pwd -P)
runs=${HABIBENCH_RUNS:-$stage/runs}
mkdir -p "$runs/$run_name"
runs=$(cd "$runs" && pwd -P)
if [[ -n "$freeze_name" && "$freeze_name" == "$run_name" ]]; then
    echo 'Use a different output name for tuning and frozen runs' >&2; exit 2
fi
oci_sha=6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9
image="vllm-openai-v0.32.0-${oci_sha}.sqfs"
revision=37561816e66b5b9153b1b64c10968c18c1e6f4a0
(
    cd "$stage/images"
    sha256sum --check "$image.sha256"
)
read -r sqfs_sha _ < "$stage/images/$image.sha256"
export HABIBENCH_OCI_DIGEST="sha256:$oci_sha" HABIBENCH_SQFS_SHA256="$sqfs_sha"
printf '%s\n%s\n' "vllm/vllm-openai@$HABIBENCH_OCI_DIGEST" "$sqfs_sha  $image" > "$runs/$run_name/container.$SLURM_JOB_ID.txt"

# Provide a job-private writable HOME/cache for compilers and GPU workers.
# The parent run directory is writable; code, input data and model snapshot are read-only.
cache="$stage/runtime/$SLURM_JOB_ID"
mkdir -p "$cache"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 VLLM_NO_USAGE_STATS=1
mounts="$repo:/habibench:ro,$data:/raw:ro,$stage/models:/models:ro,$runs:/runs:rw,$cache:/jobcache:rw"
pyxis=(--container-image="$stage/images/$image" --container-mounts="$mounts" --container-workdir=/habibench
       --container-env=HF_HUB_OFFLINE,PYTHONDONTWRITEBYTECODE,VLLM_NO_USAGE_STATS,HABIBENCH_OCI_DIGEST,HABIBENCH_SQFS_SHA256)
container_env=(env HOME=/jobcache XDG_CACHE_HOME=/jobcache/cache PYTHONPATH=/habibench)
snap="/models/$revision"
one=(srun --ntasks=1 --cpus-per-task=8 --gpus-per-task=nvidia_rtx_pro_6000:1)
if [[ "$scope" == smoke ]]; then
    "${one[@]}" "${pyxis[@]}" "${container_env[@]}" python scripts/novel/smoke.py "$snap"
    exit
fi
args=(--profile novel --data-root /raw --run-dir "/runs/$run_name" --scope "$scope")
prepare_args=("${args[@]}")
if [[ -n "$freeze_name" ]]; then prepare_args+=(--freeze-from "/runs/$freeze_name"); fi
"${one[@]}" "${pyxis[@]}" "${container_env[@]}" python -m habibench.lucas prepare "${prepare_args[@]}"
# Expand SLURM_PROCID inside the task. Each task sees its own GPU through Slurm binding.
workers=${SLURM_NTASKS:?}
srun --ntasks="$workers" --cpus-per-task=8 --gpus-per-task=nvidia_rtx_pro_6000:1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
    "${pyxis[@]}" "${container_env[@]}" bash -c \
    'exec python -m habibench.lucas worker --shard-index "$SLURM_PROCID" "$@"' _ \
    "${args[@]}" --model-path "$snap" --shard-count "$workers" --batch-size "${HABIBENCH_BATCH_SIZE:-8}"
"${one[@]}" "${pyxis[@]}" "${container_env[@]}" python -m habibench.lucas finalize "${args[@]}"
