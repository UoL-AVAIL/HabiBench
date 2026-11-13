# HabiBench

HabiBench curates UK LUCAS 2018 field photographs for paired landscape-image and Tessera training. Each view is reviewed independently. The original surveyed LC1 remains the target. A finite, 128-dimensional float32 embedding is shared by all retained views of a point.

## Run the complete 32-photo Mac test

Use Python 3.10 or newer with the extracted dataset under `data/lucas_tessera`. From the checkout root, start the Ollama app with `qwen3-vl:8b` installed, then run:

```sh
python -m pip install -e .
python -m habibench.lucas run --profile local-small
```

The first run indexes four photo associations per class and writes the selected manifest before making image requests on localhost. It reviews the 32 photos, exports the retained photo–vector pairs, verifies the export and creates an eight-photo human rating sheet. The Mac adapter uses the installed model digest. Its generation budget accommodates the thinking variant that ships under this tag. Each review sees one resized image, its view type and the broad A–H survey label. The model evaluates whether the photo is plainly out of context for that label.

The context-screen run uses `runs/local-small-context`. The local sample includes both frame-filling grassland obstructions, the grassy wetland view, duplicate close-ups, and a photo with a missing vector. Earlier classification runs and their galleries remain in their original directories. Use fresh keep/exclude ratings for this screen. Use `--data-root` or `--run-dir` for alternate locations. Repeating the command reuses successful reviews and retries errors. If the local model exhausts its token budget, the adapter retries with the same rubric and records the decoding settings of each attempt. When testing a different already-installed model, choose another run directory, for example:

```sh
python -m habibench.lucas run --profile local-small --model qwen3.8:27b --run-dir runs/local-27b-context
python -m habibench.lucas prompt --view P --class H
```

## Inspect the output and enter ratings

| Output | Contents |
| --- | --- |
| `index.jsonl` | Selected source photos, survey metadata, image hashes and vector status |
| `ledger.jsonl` | Final model response and derived decision per photo association |
| `photo-review.html` | Local gallery of decisions, reasons and original photo links |
| `identity.json`, `policy.json` | Model, settings, rubric and curation-policy provenance |
| `pilot/tune/` | Anonymised images, their survey landscape, instructions and an eight-row rating sheet |
| `curated/` | Self-contained original retained JPEGs, shared vectors, manifests and dataset card |
| `summary.json` | Decisions, verification and embedding exclusions |

For each photo, enter `keep` or `exclude` in `human_decision` and optionally add a reason. The sheet shows the survey landscape and hides Qwen’s judgement. You can reuse these eight human ratings by blind ID on the Novel tuning set. The local evaluation describes the installed 8B model.

The audit reports how many proposed exclusions human reviewers confirm, how many human-kept photos the filter removes, and how many human-identified outliers it keeps. Rates include their counts and use a null value when the denominator is zero.

```sh
cp runs/local-small-context/pilot/tune/ratings.csv my-ratings.csv
# Enter your observations in my-ratings.csv.
python -m habibench.lucas evaluate --run runs/local-small-context --labels my-ratings.csv
python -m habibench.lucas verify runs/local-small-context/curated
```

To consume a relocated export:

```python
from habibench.lucas.portable import CuratedDataset

dataset = CuratedDataset("runs/local-small-context/curated")
sample = dataset[0]
print(sample["photo_id"], sample["broad_class"], sample["image"].size, sample["embedding"].shape)
```

The reviewer keeps compatible and ambiguous landscapes, including grassy wetlands, boundary habitats and ecological close-ups. Automatic exclusion requires Qwen to identify a dominant obstruction, an extreme context failure, high evidence and no usable landscape context. A vehicle filling the frame at a grassland point can qualify. A vehicle beside an open field can still be useful. If Qwen flags a view solely because the expected habitat is missing, an observation with no obstruction stays in the dataset. Technical errors get a separate status and are retried. The training export includes each kept photo with a valid vector. The full ledger records every judgement and preserves the surveyed class. Downstream experiments choose their own splits and can group by point, photo digest and geography.

## Novel production run

Follow the [Novel runbook](scripts/novel/README.md) for pinned model and container staging, Slurm submission, the 48-photo tuning and 48-photo audit phases, and the two-GPU production pass. Model and container downloads are only needed on the staging host. The full run uses a separate identity, input manifest and output directory from the Mac test.

## Development checks

```sh
python -m pip install -e '.[test]'
python -m pytest -q
```

The A–H rubric follows Eurostat's [2018 classification](https://ec.europa.eu/eurostat/documents/205002/8072634/LUCAS2018-C3-Classification.pdf) and [instructions for surveyors](https://ec.europa.eu/eurostat/documents/205002/8072634/LUCAS-2018-C1-Instructions.pdf). The input README documents extraction at survey coordinates. The Tessera extraction year is unknown in the supplied dataset.
