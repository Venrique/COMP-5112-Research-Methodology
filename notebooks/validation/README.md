# Validation studies

`01_non_local_means.ipynb` evaluates five prespecified NonLocalMeans configurations
and an untreated baseline on 100 validation lines per dataset, sampled without
replacement with seed 42 from IAM and Bentham's predefined validation partitions.
All configurations share the same ordered samples. Source indices and dataset
identities are recorded in metadata. No test split is used.

Reusable execution and selection logic lives in `src/validation/denoising.py`.
Run cells in order using the existing inference environment. The model is loaded
once. The final cell writes the comparison plot and selection after all 12
dataset/configuration evaluations finish (1,200 line recognitions).

Within the same kernel, rerun only the evaluation cell to resume. For a fresh
kernel, restore the original source/model revisions from metadata, set OUTPUT
to the existing study folder, and change its mkdir call to exist_ok=True.
Every configuration verifies its identity before reusing predictions.

Baseline is eligible to win; best_denoising separately identifies the lowest-error
denoising configuration. Treat a subset optimum as a validation selection rather
than evidence of final test performance. The interrupted initial full-split study
is preserved separately and must not be mixed with the subset comparisons.

`02_deskew.ipynb` applies the same protocol to five deskew configurations plus
baseline. It restores source indices, IAM revision, checkpoint revision, device,
batch size and generation settings from the completed denoising study metadata.
`src/validation/deskew.py` defines the grid and reuses the existing evaluation
and selection functions. Results are under `results/validation/deskew/`.

Deskew labels expose both parameters in degrees: `deskew_max10_step0.25`
searches correction angles between -10 and +10 at 0.25-degree increments.
The five contrasts are (max_angle, angle_step): (5, .5), (10, .5), (15, .5),
(10, .25), and (10, 1). Selection keeps baseline eligible and reports the best
deskew configuration separately even if baseline wins.

`03_clahe.ipynb` repeats the same five-configuration protocol for CLAHE, using
`src/validation/clahe.py` and the shared evaluation runner. It restores the same
source indices and inference settings as the other two studies. The grid tests
clip limits 1, 2 and 4 at 8x8 tiles, plus 4x4 and 16x16 tile grids at clip limit 2.
Grid dimensions count tiles, not pixels. Labels such as `clahe_clip2_grid4x4`
expose both settings. Outputs are under `results/validation/clahe/`.

`04_grayscale.ipynb` compares the existing standard luminance conversion against
baseline on the same pinned validation samples and model. Grayscale exposes only
an on/off flag, so this is one active configuration plus baseline (four dataset/
configuration pairs, 400 recognitions), rather than five identical parameter runs.
`src/validation/grayscale.py` reuses the same evaluation and selection functions.
The notebook checks that all 100 IAM model inputs and predictions are unchanged,
because IAM is already grayscale. Results are under `results/validation/grayscale/`.

`05_intensity_normalization.ipynb` compares global min-max intensity stretching
with baseline. One shared range across all RGB channels maps to 0–255; constant
images remain unchanged. It preserves mode and size and has no numeric tuning
parameters. This differs from CLAHE's local histogram equalization.

`06_size_normalization.ipynb` independently compares heights 32, 48, 64, 96 and
128 against baseline using LANCZOS interpolation, with proportional width rounded
to the nearest integer (minimum one pixel), without cropping or padding.
TrOCR's mandatory resizing still follows this optional preprocessing.

Both use the same pinned subsets and inference settings as grayscale, and reuse
the shared runner. Their separate configuration modules are under `src/validation/`;
results go to `results/validation/intensity_normalization/` and
`results/validation/size_normalization/`. Both normalization studies have been evaluated on the pinned validation subsets.
The transforms are exported as `IntensityNormalization` and `SizeNormalization`
from `src.preprocessing`; use either alone or in an explicitly ordered
`PreprocessingPipeline`. The initial size default is 64 pixels, not a selected optimum.

The intensity-normalization notebook includes an input audit before recognition:
it verifies source integrity, output mode/dimensions, full-range/constant-image
identity and stretched endpoints, then writes `input_audit.csv` in its study folder.
The existing validation module already implements the shared pinned-sample protocol
and requires no numeric parameter grid for global min-max scaling.
