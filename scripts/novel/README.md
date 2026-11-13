# Run LUCAS photo curation on Lincoln’s Novel HPC

The Mac supplies the HabiBench code and the extracted LUCAS dataset. Novel uses a small Conda environment for staging and human evaluation. Slurm runs Qwen3.8-27B and vLLM inside the project's pinned Enroot image. All GPU jobs below are submitted from the HabiBench checkout.

The commands assume the Novel account is `hshi`, a Bash shell on Novel, and the standard shared work directory `/work/users/hshi`. Change the username and path if your account differs. Novel’s user guide covers VPN, login and storage on pp. 4–6, scheduling on pp. 9–12 and containers on pp. 14–15.

## 1. Mac — connect and create the remote directories

Off campus, connect Cisco Secure Client to `remote.lincoln.ac.uk` first. Use Novel’s login hostname and complete the Microsoft device-code/MFA prompt when requested.

```bash
cd /Users/jimmyshi/Documents/RAinLincoln/repo/HabiBench
export NOVEL=hshi@login.novel.hpc.network.uni
export NOVEL_STAGE=/work/users/hshi/habibench
ssh "$NOVEL"
```

In the **Novel** shell, run the following once. The output of `echo "$WORK"` should agree with the parent of `NOVEL_STAGE` above. `$WORK` is shared by login and compute nodes and has no backup.

```bash
echo "$WORK"
lsquota
df -h "$WORK"
mkdir -p "$HOME/repos" "$WORK/habibench/data/lucas_tessera" "$WORK/habibench/runs"
exit
```

The raw dataset occupies about 24 GiB, the BF16 weights about 52 GiB and the curated export can approach another 24 GiB. Allow additional working space for the vLLM image, compressed and unpacked layers, Conda and a temporary second export during replacement.

## 2. Mac — transfer the extracted data

```bash
rsync -rtP --exclude='*.zip' --exclude='.DS_Store' \
  data/lucas_tessera/ "$NOVEL:$NOVEL_STAGE/data/lucas_tessera/"
```

The trailing slash copies the contents into the destination. This sends about 23.75 GiB, including the survey table, habitat lookup, JPEGs and Tessera files. The ZIPs in the same directory contain duplicate copies. If the connection drops, repeat the same command. To check file contents after transfer, the following dry run reads both copies and lists any files that still differ. Repeat the transfer with `-c` if it lists files.

```bash
rsync -rcni --exclude='*.zip' --exclude='.DS_Store' \
  data/lucas_tessera/ "$NOVEL:$NOVEL_STAGE/data/lucas_tessera/"
```

## 3. Mac → Novel — transfer the code with Git

At the time this guide was prepared, this Mac folder contained the source and had no Git repository or configured remote. The **Git bundle** route works over your existing Novel SSH connection and requires no Git hosting account. On the **Mac**, in the project root:

```bash
git init -b main
git add .gitignore README.md pyproject.toml habibench/ scripts/ tests/
git diff --cached --stat
git commit -m "Add HabiBench LUCAS context curation pipeline"
git bundle create /tmp/HabiBench-main.bundle main
rsync -tP /tmp/HabiBench-main.bundle "$NOVEL:repos/HabiBench-main.bundle"
```

This selection includes the curation code, tests and launcher. The dataset, local runs and other project documents are outside the commit. On **Novel**, clone the bundle:

```bash
ssh "$NOVEL"       # run this line on the Mac
```

```bash
git clone -b main "$HOME/repos/HabiBench-main.bundle" "$HOME/repos/HabiBench"
cd "$HOME/repos/HabiBench"
git rev-parse --short HEAD
```

**Hosted Git option.** If you prefer GitHub or GitLab, create an empty private repository, use the same selective commit, then run `git remote add origin YOUR_REPO_URL` and `git push -u origin main` on the Mac. On Novel use `git clone -b main YOUR_REPO_URL "$HOME/repos/HabiBench"` in place of cloning the bundle. To update either checkout later, commit on the Mac, rebuild and transfer the bundle or push to the hosted remote, then run `git pull --ff-only` from the Novel checkout between runs.

## 4. Novel — set the paths and create the Conda environment

Keep a small setup file so the same paths are available when you reconnect. The following commands create it under the project's work directory.

```bash
cat > "$WORK/habibench/paths.sh" <<'PATHS'
export HABIBENCH_REPO="$HOME/repos/HabiBench"
export HABIBENCH_STAGE="$WORK/habibench"
export HABIBENCH_DATA="$HABIBENCH_STAGE/data/lucas_tessera"
export HABIBENCH_RUNS="$HABIBENCH_STAGE/runs"
export HF_HOME="$HABIBENCH_STAGE/hf-cache"
export CONDA_PKGS_DIRS="$HABIBENCH_STAGE/conda-pkgs"
PATHS
source "$WORK/habibench/paths.sh"
cd "$HABIBENCH_REPO"
ls "$HABIBENCH_DATA"/{UK_2018_20200213.csv,habitats.csv}
find "$HABIBENCH_DATA/lucas-dataset" -type f -iname '*.jpg' | wc -l
find "$HABIBENCH_DATA/geospatial" -type f -name '*.npz' | wc -l
```

Expect **63,135 JPEGs and 11,153 NPZ files**, including 66 placeholder vectors. The pipeline checks vector validity separately.

If `conda --version` already works, use that installation. Otherwise run `module avail` and load the exact Conda, Miniforge or Miniconda module listed for your account. The site guide does not specify its name. If no module is provided and user installations are allowed, an alternative for a network-enabled Linux x86_64 host is:

```bash
curl -fL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
  -o "$HABIBENCH_STAGE/Miniforge3-Linux-x86_64.sh"
bash "$HABIBENCH_STAGE/Miniforge3-Linux-x86_64.sh" -b -p "$HABIBENCH_STAGE/miniforge3"
source "$HABIBENCH_STAGE/miniforge3/etc/profile.d/conda.sh"
```

Create the environment once. The GPU runtime is supplied by the container.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y --prefix "$HABIBENCH_STAGE/conda-env" python=3.11 pip
conda activate "$HABIBENCH_STAGE/conda-env"
python -m pip install -e '.[staging]'
python -m habibench.lucas prompt --view W --class H
```

On **subsequent logins**, source `"$WORK/habibench/paths.sh"`, reload the same Conda module or source the Miniforge initialization file above, activate the existing environment and `cd "$HABIBENCH_REPO"`. [Conda environments](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html) · [Miniforge installer](https://github.com/conda-forge/miniforge)

## 5. Novel staging — download Qwen and the inference container once

Use a network-enabled Linux x86_64 staging host with Enroot. The host needs access to Docker Hub and Hugging Face. From the activated Conda environment:

```bash
cd "$HABIBENCH_REPO"
command -v enroot
command -v flock
command -v sha256sum
bash scripts/novel/stage.sh
```

The script first imports the pinned vLLM 0.32.0 image, then downloads the BF16 weights, processor, tokenizer and chat template for `Qwen/Qwen3.8-27B`. It verifies the snapshot and writes the revision marker that the worker checks. After staging, confirm both outputs:

```bash
rev=37561816e66b5b9153b1b64c10968c18c1e6f4a0
cat "$HABIBENCH_STAGE/models/$rev/HABIBENCH_REVISION"
ls -lh "$HABIBENCH_STAGE/images/"*.sqfs
```

If Novel restricts outbound downloads or Enroot imports on the login node, use the site-designated staging host. If that host has a different filesystem, run the same stage script from the same checkout there, setting `HABIBENCH_STAGE` to a writable directory; then transfer its `models/` and `images/` directories, including the checksum sidecar, to the same directories under `/work/users/hshi/habibench/`. The GPU workers load the staged files offline. [Qwen model](https://huggingface.co/Qwen/Qwen3.8-27B) · [Pinned revision](https://huggingface.co/api/models/Qwen/Qwen3.8-27B/revision/37561816e66b5b9153b1b64c10968c18c1e6f4a0) · [Enroot import](https://github.com/NVIDIA/enroot/blob/main/doc/cmd/import.md)

## 6. Novel — run one real photo, then the 48-photo tuning set

Submit from the checkout with the path variables exported. The smoke job loads the model on one RTX PRO 6000 and reviews one woodland photo. Slurm prints the job ID.

```bash
cd "$HABIBENCH_REPO"
sbatch --ntasks=1 --gpus=nvidia_rtx_pro_6000:1 --mem=64G --time=01:00:00 \
  scripts/novel/review.sbatch smoke
squeue -u "$USER"
```

After the job starts, inspect `lucas-qwen-JOBID.log` in the checkout, substituting the number printed by `sbatch`. Look for an `observation` containing `context`, `obstruction` and `reason`. After the smoke job succeeds, submit tuning:

```bash
sbatch --ntasks=1 --gpus=nvidia_rtx_pro_6000:1 --mem=64G --time=02:00:00 \
  scripts/novel/review.sbatch tune novel-context-tune-v1
```

After completion, check `"$HABIBENCH_RUNS/novel-context-tune-v1/summary.json"` for 48 reviews and zero errors. The photo sheet and images are in `pilot/tune/` under that run.

## 7. Rate the pilot, then run and rate the held-out audit

For each image, use its broad survey label and fill in `human_decision` with `keep` or `exclude`; `reason` is optional. Rate before looking at Qwen's decisions. Eight previously completed ratings from the **context-screen** Mac pilot can count toward the 48 tuning ratings when matched by `blind_id`. Rate all 48 held-out audit photos independently.

To rate the tuning set on your **Mac**, reuse the Mac variables from step 1. Download the whole folder so the relative image paths resolve, open the CSV, and save your edited copy under the name below. Keep the header and the blind IDs intact.

```bash
mkdir -p "$HOME/Downloads/habibench-review/tune"
rsync -rtP "$NOVEL:$NOVEL_STAGE/runs/novel-context-tune-v1/pilot/tune/" \
  "$HOME/Downloads/habibench-review/tune/"
cp "$HOME/Downloads/habibench-review/tune/ratings.csv" "$HOME/Downloads/habibench-review/tune/tuning-ratings.csv"
# Edit tuning-ratings.csv now. When finished:
rsync -tP "$HOME/Downloads/habibench-review/tune/tuning-ratings.csv" "$NOVEL:$NOVEL_STAGE/runs/"
```

On **Novel**, score your ratings and submit the held-out run using the accepted tuning configuration. If you change the rubric after tuning, use a fresh tuning run name and reference that name in subsequent jobs.

```bash
python -m habibench.lucas evaluate --run "$HABIBENCH_RUNS/novel-context-tune-v1" \
  --labels "$HABIBENCH_RUNS/tuning-ratings.csv" --split tune
sbatch --ntasks=1 --gpus=nvidia_rtx_pro_6000:1 --mem=64G --time=02:00:00 \
  scripts/novel/review.sbatch audit novel-context-audit-v1 novel-context-tune-v1
```

After the audit job completes, on the **Mac**:

```bash
mkdir -p "$HOME/Downloads/habibench-review/audit"
rsync -rtP "$NOVEL:$NOVEL_STAGE/runs/novel-context-audit-v1/pilot/audit/" \
  "$HOME/Downloads/habibench-review/audit/"
cp "$HOME/Downloads/habibench-review/audit/ratings.csv" "$HOME/Downloads/habibench-review/audit/audit-ratings.csv"
# Edit audit-ratings.csv now. When finished:
rsync -tP "$HOME/Downloads/habibench-review/audit/audit-ratings.csv" "$NOVEL:$NOVEL_STAGE/runs/"
```

On **Novel**, review the audit results. Look at how many exclusions humans confirmed and how many usable photos the filter removed. The report includes denominators and inference coverage.

```bash
python -m habibench.lucas evaluate --run "$HABIBENCH_RUNS/novel-context-audit-v1" \
  --labels "$HABIBENCH_RUNS/audit-ratings.csv" --split audit
```

## 8. Novel — run all photos, monitor and resume

```bash
cd "$HABIBENCH_REPO"
sbatch scripts/novel/review.sbatch all novel-context-full-v1 novel-context-tune-v1
squeue -u "$USER"
```

The job requests two RTX PRO 6000 96 GB GPUs on one node for up to 12 hours. Each GPU reviews its own shard with a separate 27B model replica. The time limit is an allocation request. Read `timing-0.json` and `timing-1.json` in the full run after each worker has timed about 250 photos; the larger `projected_shard_hours` estimates two-GPU inference time. Slurm queue, indexing, loading and export take additional time. To follow an active job use `tail -f lucas-qwen-JOBID.log` from the checkout.

If time runs out or there are retriable errors, resubmit **the same full-run command** after the previous job ends. The two journals preserve successful reviews. Keep the run name, tuning reference and shard count the same.

## 9. Novel → Mac — verify and save the portable release

After the full job finishes, check `summary.json` for 63,135 reviewed photos and zero inference errors. Review the kept photos without valid vectors separately. Attach the completed human reports, refresh the export and verify its image–vector pairs in the Conda environment. Export verification reads all retained JPEGs; use CPU job time for this step if required by site policy.

```bash
full="$HABIBENCH_RUNS/novel-context-full-v1"
cat "$full/summary.json"
python -m habibench.lucas attach-evaluation --run "$full" \
  --source-run "$HABIBENCH_RUNS/novel-context-tune-v1" --split tune
python -m habibench.lucas attach-evaluation --run "$full" \
  --source-run "$HABIBENCH_RUNS/novel-context-audit-v1" --split audit
python -m habibench.lucas finalize --profile novel --scope all --data-root "$HABIBENCH_DATA" --run-dir "$full"
python -m habibench.lucas verify "$full/curated"
```

The training dataset is `$full/curated/`. Its `photos.jsonl` lists retained JPEG–vector pairs, `points.jsonl` lists points and their views, `excluded.jsonl` lists excluded photos, and `retention.json` breaks decisions down by class and view. To back up the release on the **Mac**, from the HabiBench root:

```bash
mkdir -p runs/novel-context-full-v1/curated
rsync -rtP "$NOVEL:$NOVEL_STAGE/runs/novel-context-full-v1/curated/" runs/novel-context-full-v1/curated/
python -m habibench.lucas verify runs/novel-context-full-v1/curated
```

Keep the two completed human rating sheets and the Git commit alongside your project backups. The portable dataset contains the evaluation reports.
