# HTR preprocessing experiments

This research project investigates how optional image preprocessing affects
pretrained handwritten text recognition (HTR) models. See
[RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the methodology.

Currently implemented: IAM dataset loading, a configurable preprocessing
pipeline, non-local means denoising, and a visual comparison notebook.
Model inference and CER/WER evaluation are not yet implemented.

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
accuracy improvements. The next development priority is pretrained TrOCR
inference with aggregate CER/WER evaluation.
