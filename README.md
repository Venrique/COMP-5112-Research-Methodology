# HTR preprocessing experiments

## Bentham dataset

[Bentham R0](https://huggingface.co/datasets/staghado/Bentham) is stored locally
under `data/bentham/`, which is ignored by Git. The repository distributes raw
archives, so use the local loader rather than Hugging Face `load_dataset`.
To reproduce the download (about 3.5 GB; allow about 9 GB including extraction):

```sh
.venv/bin/python -m src.load_dataset.bentham --download
```

The downloader pins commit `bfd99574602e2abbe7896255b1fd5e114f68571a`,
verifies archive SHA-256 checksums, safely extracts both archives, and records
`source.json`. No additional dependencies are needed beyond the existing setup.
From a notebook with the project root on `sys.path`:

```python
from src.load_dataset.bentham import load_bentham_dataset, get_sample

dataset = load_bentham_dataset(split="validation")
sample = get_sample(dataset, 0)
display(sample["image"])
print(sample["ground_truth"])
```

The Dataset provides `id`, `image`, and `text`, supports indexing and `select`,
and lazily decodes original line images, preserving their modes (including RGBA).
For preprocessing steps requiring L/RGB, composite transparency onto white and
convert to RGB consistently for baseline and every treatment; preserve originals.
Its default path works from
notebook subdirectories; pass `data_dir` to use another location. Supplied
train/validation/test line partitions and their order are preserved. Use validation
for development and reserve test for final evaluation. Transcriptions retain
punctuation spacing; only the terminal newline is removed. Images are paired
with the supplied line transcripts, not PAGE XML IDs, which may differ.
Record the source commit and split in experiments. The Hugging Face repository
does not declare a license; verify usage terms before redistributing the data.

Experiment A [Part 3](notebooks/experiments/experiment_a_part_3.ipynb) compares
baseline and the Part 2 order (grayscale → denoising → deskewing → CLAHE) with
grayscale → CLAHE → denoising → deskewing. All three are freshly evaluated on
the same default 100 seeded validation lines with fixed parameters and pinned
dataset/model revisions. The notebook saves corpus and paired-line comparisons,
direct changes versus Part 2, cumulative image-stage previews, predictions, and
metadata in a fresh `results/experiment_a_part_3_*` folder. This tests order;
it does not establish that smoother-looking backgrounds improve recognition.

## Experiment A: individual techniques

Run [experiment_a.ipynb](notebooks/experiments/experiment_a.ipynb) from top to
bottom in the project environment. It compares baseline, grayscale, denoising,
deskewing, and CLAHE with the existing TrOCR wrapper and corpus CER/WER evaluator.
The default selects 100 IAM validation lines without replacement using seed 5112;
set `SAMPLE_COUNT = None` for the full validation split. Dataset and model commits
are pinned in the configuration cell. The test split remains reserved.

Each run writes predictions, summaries, sample information, metadata, and PNG
charts to a fresh `results/experiment_a_*` directory. Plots show corpus errors,
changes from baseline, paired line errors, outcome counts, and selected image and
prediction examples. Grayscale is explicitly identified as redundant when all
selected inputs are already grayscale. These are development comparisons, not
final test results or significance claims. No additional dependencies are needed.

Earlier demonstration notebooks now live in `notebooks/demo/`.

### Experiment A, Part 2: all techniques together

Run [experiment_a_part_2.ipynb](notebooks/experiments/experiment_a_part_2.ipynb)
to compare baseline against grayscale → denoising → deskewing → CLAHE. It uses
the same default samples, pinned revisions, parameters, and decoding as Part 1.
The order is explicit and configurable in the pipeline tuple. This evaluates
one combined treatment, without attributing its effect to an individual step.

Fresh `results/experiment_a_part_2_*` directories contain predictions, corpus
summaries, sample information, metadata, and charts: cumulative image stages,
corpus errors and changes, observed timing, paired line errors, outcome counts,
and selected image/prediction examples. Set `SAMPLE_COUNT = None` for the full
validation split. The test split remains reserved.

## Your own handwriting images

Use `notebooks/05_local_images_recognition.ipynb` for local images. Existing
IAM notebooks are unchanged. Put one English handwritten line per image in
`data/local/images/`, then edit `templates/local_images.csv`. Replace the example
filename and add one row per image. Image paths are relative to the CSV location;
quote ground-truth fields containing commas. Blank ground truth enables prediction
without scoring; fill it in later and rerun the notebook to obtain CER/WER.
Only listed images are used. The template's example image must be supplied or
its row replaced before running. Local images under `data/` are git-ignored;
the template is tracked, so avoid committing private transcriptions to it.

The notebook compares baseline, denoising and deskewing individually using the
existing preprocessing pipeline. Set `MANIFEST`, `SPLIT`, model and pipeline
settings in its configuration cell. Use validation for development, and reserve
test images for final evaluation. Full-page images require line cropping first.
EXIF orientation, white transparency compositing and RGB decoding apply uniformly
to all configurations without overwriting originals. Metrics use only labeled
samples and report both total and scored counts. All-unlabeled runs have no
CER/WER. Exports include predictions, summaries, image hashes, manifest contents,
model revision and environment metadata in a fresh `results/local_trocr_*` folder.

No additional dependencies beyond `requirements-inference.txt` are needed.

This research project investigates how optional image preprocessing affects
pretrained handwritten text recognition (HTR) models. See
[RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the methodology.

Currently implemented: IAM dataset loading, a configurable preprocessing
pipeline with CLAHE, non-local means denoising, and deskewing; TrOCR inference;
and corpus CER/WER evaluation.

## TrOCR recognition comparison

Install `requirements-inference.txt` into the existing virtual environment, then
open [03_trocr_recognition.ipynb](notebooks/03_trocr_recognition.ipynb) and run
top to bottom. The checkpoint is downloaded to the Hugging Face cache on first
use. The notebook compares baseline and non-local means denoising on the same
10 validation samples and shows images, raw predictions, corpus CER/WER and
changes relative to baseline. It saves predictions.csv, summary.csv and
metadata.json in a fresh timestamped directory under results/.

Configure checkpoint/revision, device, decoding and sample indices in the
configuration cell. CPU is the verified default; CUDA/MPS are not tested.
Model and dataset revisions are resolved to immutable commits and recorded.
For an exact rerun, set the revision fields to the commits in metadata.json.
Raw text is preserved; scoring collapses whitespace and retains case and
punctuation. Rates are fractions, may exceed 1, and aggregate corpus edit
counts rather than averaging per-line scores. IAM punctuation spacing can
therefore contribute to the measured errors.

Reusable inference lives in `src/models/`; evaluation lives in `src/evaluation/`.
To use another architecture, implement `predict(image) -> str` and
`to_config() -> dict` as described by `HandwritingRecognizer`, then pass the
instance to `compare_recognition`. Keep mandatory model input transforms inside
the wrapper. Add preprocessing configurations to the notebook's `pipelines`
dictionary without changing recognition or metric code. No training is performed.

## Setup

Use the existing project Python environment. From the repository root, install
the preprocessing and demonstration dependencies:

```sh
.venv/bin/python -m pip install -r requirements-preprocessing.txt
```

Open notebooks in PyCharm or another Jupyter-capable editor and select the
project environment as the kernel. The requirements file includes the libraries
used by the demonstration, but does not install a standalone Jupyter server.
Loading IAM may require network access to Hugging Face on the first run.

## Enable or disable preprocessing

Run Python examples from the repository root so that `src` is importable.

```python
from src.preprocessing import NonLocalMeans, PreprocessingPipeline

# Baseline: no optional preprocessing; returns a pixel-identical image copy.
baseline = PreprocessingPipeline()

# Enable non-local means denoising.
denoising = PreprocessingPipeline((
    NonLocalMeans(enabled=True, h=3.0, h_color=3.0),
))

# Keep the step configured but skip its execution.
disabled = PreprocessingPipeline((
    NonLocalMeans(enabled=False, h=3.0, h_color=3.0),
))
```

The comma in `(NonLocalMeans(...),)` makes it a one-element tuple of steps.
The pipeline applies enabled steps in tuple order and skips disabled steps.
An empty pipeline and a pipeline with every step disabled both preserve the
input pixels. All return a separate image, leaving the original unchanged.

Pipeline settings are immutable. To change a parameter or enabled flag,
construct a new step/pipeline; in a notebook, edit and rerun the configuration
cell, then rerun the comparison cells. Create each pipeline once and reuse it
across samples.

### Run preprocessing on an image from Git Bash

From the project root, run the script with input and output paths. By default,
it applies CLAHE, non-local means denoising, then deskewing:

```sh
./.venv/Scripts/python.exe run_preprocessing.py input.png results/processed.png
```

Choose individual steps and their order with `--steps`:

```sh
./.venv/Scripts/python.exe run_preprocessing.py input.png results/clahe.png --steps clahe
./.venv/Scripts/python.exe run_preprocessing.py input.png results/baseline.png --steps
```

The script saves the processed image and prints its preprocessing configuration.
It converts unsupported input modes (such as palette or RGBA) to RGB; `L` and
`RGB` images retain their mode. These command-line runs process images only and
do not calculate HTR recognition accuracy.

### Apply pipelines to the same sample

```python
from src.load_dataset.iam import load_iam_dataset, get_sample

# Always specify validation for development; the existing loader defaults to test.
dataset = load_iam_dataset(split="validation")
sample = get_sample(dataset, 0)

original_image = sample["image"]
baseline_image = baseline(original_image)
denoised_image = denoising(original_image)
unchanged_image = disabled(original_image)
print(sample["ground_truth"])
```

Images must be Pillow images. Enabled denoising supports 8-bit grayscale (`L`)
and RGB modes, retaining mode and dimensions. It rejects other modes rather
than silently converting them. No grayscale conversion, resizing, or text
normalization is added by this pipeline.

## Configure non-local means denoising

Non-local means compares similar image patches to reduce noise. The
implementation uses OpenCV's fast non-local means routines.

| Parameter | Default | Purpose |
| --- | --- | --- |
| `enabled` | `True` | Whether the step runs |
| `h` | `3.0` | Grayscale/luminance smoothing strength |
| `h_color` | `3.0` | Color smoothing strength for RGB images; unused for `L` |
| `template_window_size` | `7` | Size of patches compared |
| `search_window_size` | `21` | Size of the area searched for similar patches |

Strengths must be finite and positive. Window sizes must be positive odd
integers, and the search window must be at least as large as the template window.
Parameters are validated even when the step is disabled.

Higher strengths can remove fine handwriting strokes. Larger search windows
increase computation. The defaults are starting values, not experimentally
selected settings. Clean images may show very little visible change at these
settings; visible smoothing alone does not establish better recognition.

## Run the side-by-side demonstration

Open [02_denoising_demo.ipynb](notebooks/02_denoising_demo.ipynb) and run its
cells from top to bottom. It loads IAM's validation split and compares three
samples under the baseline and denoising configurations, with transcriptions,
preprocessing timings, and configuration metadata.

In its configuration cell:

- Change `SAMPLE_INDICES` to inspect other validation samples.
- Set `enabled=False` to bypass denoising.
- Adjust the denoising parameters to compare settings.
- Add entries to `pipelines` to display additional comparison columns.

For example, replace the notebook's `pipelines` dictionary with:

```python
pipelines = {
    "baseline": PreprocessingPipeline(),
    "denoising_h3": PreprocessingPipeline((
        NonLocalMeans(h=3.0, h_color=3.0),
    )),
    "denoising_h8": PreprocessingPipeline((
        NonLocalMeans(h=8.0, h_color=8.0),
    )),
}
```

Rerun the configuration, data/metadata, and plotting cells after changes.
The earlier [dataset exploration notebook](notebooks/01_dataset_exploration.ipynb)
demonstrates loading and displaying samples, but uses the test split.
Use the validation demonstration for preprocessing development.

## Record experiment configurations

```python
import json

configuration = {
    name: pipeline.to_config()
    for name, pipeline in pipelines.items()
}
print(json.dumps(configuration, indent=2))
```

`to_config()` records steps in their configured order, including disabled steps
and all their parameters. It does not execute or reconstruct a pipeline.
Save this configuration with eventual experiment results.

For fair recognition comparisons, use identical samples and ground truth for
every configuration, fix the checkpoint and decoding settings, and apply the
model's mandatory input transforms consistently after optional preprocessing.
Report aggregate CER/WER. Reserve the test set for final evaluation.

The demonstration records library versions and a dataset fingerprint. The
existing IAM loader does not pin a dataset revision; a fingerprint alone is not
a substitute for revision pinning in final reproducible experiments.

## Add future techniques

Denoising is currently the only implemented technique. A new step should follow
`PreprocessingStep` in `src/preprocessing/pipeline.py`: expose an `enabled` flag,
accept and return a Pillow image through `__call__`, and return JSON-serializable
settings through `to_config()`. Preserve its input image. Add the step to the
pipeline tuple in the desired execution order; each technique can then be
enabled independently. Test individual techniques before combinations.

## Project files and tests

| Path | Purpose |
| --- | --- |
| `src/load_dataset/iam.py` | IAM loading and sample extraction |
| `src/preprocessing/pipeline.py` | Denoising and ordered pipeline |
| `notebooks/02_denoising_demo.ipynb` | Validation-image comparisons |
| `tests/test_preprocessing.py` | Synthetic preprocessing checks |
| `requirements-preprocessing.txt` | Pinned demonstration dependencies |

Run tests from the repository root:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover baseline preservation, disabled steps, execution order, input
preservation, parameter validation, RGB channel handling, configuration
serialization, and synthetic noise reduction. They do not establish HTR
accuracy improvements. Recognition tests additionally check known corpus edit
counts, normalization and interchangeable model evaluation. The next development
priority is a larger fixed validation comparison before adding techniques.

## CNN + LSTM + CTC

### Final IAM baseline result

**Training finished through early stopping at epoch 17 of the requested maximum
20. The selected model is epoch 14: validation CER 10.07%, WER 34.29%.**
These scores use the fixed 100 IAM validation lines. CNN-LSTM-CTC test evaluation
and optional preprocessing comparisons have not been run.

Start with the [complete implementation and research report](docs/CNN_LSTM_CTC.md)
or the [short final run summary](results/cnn_lstm_ctc/training/20261114_baseline/iam/FINAL_SUMMARY.md).
All CNN-LSTM-CTC documentation and artifact links are grouped here; the existing
TrOCR implementation and results are retained.

| Item | Value |
| --- | --- |
| Full training scope | IAM only; original 6,482 training lines |
| Validation | Same saved 100 lines from the original 976-line validation split |
| Test | Same saved 250 IAM test indices reserved for later final comparison |
| Architecture | Three CNN blocks, two-layer bidirectional LSTM, classifier, CTC |
| Parameters / classes | 772,688 parameters; 79 training-derived characters plus blank |
| Input | Height-64 grayscale, aspect ratio preserved, pixels [0,1], right white padding |
| Training | Adam 0.001; batch 4; seed 5112; clip norm 5; four CPU threads |
| Stopping/selection | Patience 3; lowest validation CER, then WER, then validation loss |
| Selected checkpoint | `best.pt`, epoch 14: CER 10.07%, WER 34.29% |
| Latest checkpoint | `last.pt`, epoch 17: CER 10.61%, WER 36.54% |

Epoch 14 has the best CER; epoch 13 has the best WER (34.19%), and epoch 16
has the lowest validation loss. Epochs 15-17 did not beat epoch 14's selection
score. No epoch 18-20 results exist. CER/WER are error rates, not simple accuracy
percentages. Do not compare these validation scores directly with TrOCR's test
scores or infer architecture effects without matched samples/protocols.

Bentham preparation, model checks, and small diagnostic runs are preserved,
but full Bentham CNN-LSTM-CTC training was excluded at the user's request.

### Notebooks and reusable code

| File | Purpose |
| --- | --- |
| [01_data_preparation.ipynb](notebooks/cnn_lstm_ctc/01_data_preparation.ipynb) | Original/prepared images, real text-to-ID examples, split counts, training text-length graphs |
| [02_model_check.ipynb](notebooks/cnn_lstm_ctc/02_model_check.ipynb) | Untrained forward/CTC-loss/backward checks and synthetic repeated-letter decoding |
| [03_training.ipynb](notebooks/cnn_lstm_ctc/03_training.ipynb) | Final status, all completed epochs, loss/error graphs, checkpoint usage; training launch disabled by default |
| [Notebook guide](notebooks/cnn_lstm_ctc/README.md) | How to open and understand the workflow |
| [ctc_data.py](src/training/ctc_data.py) | Training-only vocabulary, mandatory image preparation, split wrapper, batching |
| [cnn_lstm_ctc.py](src/models/cnn_lstm_ctc.py) | CNN, BiLSTM, classifier, CTC loss, greedy decoder, recognizer adapter |
| [ctc_training.py](src/training/ctc_training.py) | Alignment audits, training, validation selection, early stopping, checkpoints, resume, plots, CLI |
| [epoch_reporting.py](src/training/epoch_reporting.py) | Verifies saved metrics/cohorts, documents epochs, archives available weights |
| [Model exports](src/models/__init__.py) | Exposes CNN-LSTM-CTC alongside TrOCR |
| [Training package](src/training/__init__.py) | Training package initialization |

Character IDs come from training text only; blank ID 0 is never a target letter.
Original images/text and existing splits are preserved. Baseline training uses
mandatory preparation only, with no optional denoising, deskewing, CLAHE,
intensity normalization, or experimental size normalization. Grayscale is
mandatory here, so a separate optional grayscale treatment can be redundant.

### Results, predictions, and example tables

The full run is in `results/cnn_lstm_ctc/training/20261114_baseline/iam/`.

- [Every completed epoch with fixed prediction examples](results/cnn_lstm_ctc/training/20261114_baseline/iam/EPOCH_RESULTS.md)
- [Full-precision analysis table](results/cnn_lstm_ctc/training/20261114_baseline/iam/epoch_summary.csv)
- [Original epoch history](results/cnn_lstm_ctc/training/20261114_baseline/iam/history.csv)
- [Training/validation loss and error graphs](results/cnn_lstm_ctc/training/20261114_baseline/iam/training_curves.png)
- [Original epoch 14 predictions](results/cnn_lstm_ctc/training/20261114_baseline/iam/validation_epoch_014.csv)
- [100 epoch 14 examples: readable table](results/cnn_lstm_ctc/training/20261114_baseline/iam/examples/epoch_014_random_100_seed5112.md)
- [100-example CSV](results/cnn_lstm_ctc/training/20261114_baseline/iam/examples/epoch_014_random_100_seed5112.csv)
- [Random-order selection manifest](results/cnn_lstm_ctc/training/20261114_baseline/iam/examples/epoch_014_random_100_seed5112.json)
- [Final status](results/cnn_lstm_ctc/training/20261114_baseline/iam/status.json) and [original signed configuration](results/cnn_lstm_ctc/training/20261114_baseline/iam/metadata.json)
- [Final artifact verification](results/cnn_lstm_ctc/training/20261114_baseline/iam/FINAL_VERIFICATION.json)

The random-order export contains **all the same 100 validation lines**, shuffled
without replacement using seed 5112. It is not new inference, new samples, or a
test evaluation. `ground_truth` is the actual text; `prediction` is the model's
output. The epoch report links all 17 original prediction CSVs.

Checkpoint snapshots for epochs 2-17 are in `epoch_checkpoints/`; epoch 1's
weights were overwritten before archiving began, but its scores and predictions
remain. `.pt` files are binary and Git-ignored: preserve them separately with
recorded hashes. `training_smoke/` contains earlier 16-example diagnostics,
not the full run. Preparation/model-check artifacts are also stored separately.
The full report explains every result folder, audit, and supervisor/request file.

### Load the selected model

Use the existing project Python environment and run from the project root:

```python
from src.models import CNNLSTMCTCRecognizer

recognizer = CNNLSTMCTCRecognizer.from_checkpoint(
    "results/cnn_lstm_ctc/training/20261114_baseline/iam/best.pt",
    device="cpu",
)
# prediction = recognizer.predict(a_pillow_line_image)
# settings = recognizer.to_config()  # includes checkpoint SHA-256 and epoch
```

Loading restores matching character IDs and mandatory input preparation. The
training launcher in the notebook stays disabled so opening it does not train
again. Epoch-boundary resume preserves optimizer/shuffle/RNG state and checks
run identity. **This run has already early-stopped with patience counter 3:**
increasing only the epoch limit will not reset patience or start epoch 18.
Further training needs a deliberate, separately documented continuation policy;
none was started by this documentation update.

### Verification and next research task

The selected checkpoint was also run on all 10 images in `data/local/`, using
the existing `templates/local_images.csv` labels unchanged. This separate local
diagnostic scored **28.10% CER and 78.67% WER**. It is not IAM test performance.
No training, optional preprocessing, or spelling correction was applied; source
images, manifest, and checkpoint were verified unchanged.

- [Executed local-image notebook](notebooks/cnn_lstm_ctc/04_local_images_recognition.ipynb)
- [Actual versus predicted text and image preview](results/cnn_lstm_ctc/local_predictions/20261115T054718447180Z/PREDICTIONS.md)
- [Local predictions CSV](results/cnn_lstm_ctc/local_predictions/20261115T054718447180Z/predictions.csv)

The notebook reuses the existing local-image loader, recognizer, baseline
pipeline, and corpus evaluator; rerunning writes a fresh result directory.

The implementation suite previously passed 59 tests. Tests cover data/label
preservation, CTC targets and repeated letters, model gradients, padding
consistency, checkpoint reload, exact resume, early stopping, invalid inputs,
and reporting consistency:

- [Data tests](tests/test_ctc_data.py)
- [Model tests](tests/test_cnn_lstm_ctc.py)
- [Training tests](tests/test_ctc_training.py)
- [Epoch-report tests](tests/test_epoch_reporting.py)

Final artifact checks verified all 17 epoch scores against 1,700 saved
predictions, identical validation indices/ground truth, all 16 archived hashes,
best/last checkpoint identities, and the 100-example export. These checks do
not establish final test performance. GPU execution remains untested.

Next: freeze epoch 14, select optional preprocessing on validation, define the
individual/combined/order comparisons, then use the same 250 IAM test lines as
TrOCR for final evaluation. Keep mandatory input preparation fixed and disclose
the models' different training provenance. See the full report for all 17 epoch
results, interpretation, limitations, and reproducibility details.
