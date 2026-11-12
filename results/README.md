# Research results

New studies use `results/<phase>/<technique>/<UTC-study-id>/`.

- `validation/denoising/`: parameter selection on predefined validation splits.
- Reserve `test/` for final evaluations after selecting and freezing parameters.
- Existing top-level study directories are retained to preserve notebook references.

A denoising study contains `metadata.json`, `comparison.csv`,
`equal_dataset_ranking.csv`, `dataset_winners.csv`, and a comparison plot.
Each `<dataset>/<configuration>/` contains a resume identity, original ground
truth and predictions in `predictions.csv`, and corpus metrics in `summary.json`.
Partial studies contain only completed batches/configurations; do not interpret
an incomplete ranking as a final choice. CER/WER are ratios (multiply by 100 for
percentages); negative baseline deltas mean improvement.

Deskew studies use `validation/deskew/<UTC-study-id>/` and the same per-dataset,
per-configuration layout. `selection.json` distinguishes `best_overall`
(including baseline) from `best_deskew`. A complete study has a `status.json`
with 12 finished dataset/configuration pairs: five deskew settings plus baseline
on each dataset. The metadata references the denoising study used to pin samples
and inference settings.

CLAHE studies use `validation/clahe/<UTC-study-id>/` with the same layout and
12 dataset/configuration pairs. `selection.json` records `best_overall`
(including baseline) and `best_clahe` separately, plus selected parameters and
per-dataset winners. The metadata pins the original denoising study's samples
and inference settings for a controlled comparison across techniques.

Grayscale studies use `validation/grayscale/<UTC-study-id>/`. Standard grayscale
has no numeric tuning parameters, so completion means four dataset/configuration
pairs: baseline and grayscale on both datasets. The notebook verifies the IAM
identity control, then saves the same predictions, comparison tables, plot,
selection and completion status as the other studies. Exact score ties do not
establish an accuracy benefit from enabling grayscale.

Intensity normalization studies use `validation/intensity_normalization/<UTC-study-id>/`.
The fixed global min-max algorithm has one active configuration plus baseline,
so a complete study contains four dataset/configuration pairs. `input_audit.csv`
records each sample's before/after ranges and whether pixel values changed;
full-range and constant images are unchanged by design.

Size normalization studies use `validation/size_normalization/<UTC-study-id>/`.
A complete study has 12 dataset/configuration pairs: five target heights plus
baseline on each dataset. Selection records the best optional resize separately
from the overall winner. TrOCR's mandatory processor resize is held fixed.

Final Experiment A runs use `test/experiment_a/<UTC-run-id>/`: 250 fixed test
lines per dataset, six individual techniques plus baseline. The finalized
protocol selects each technique's active configuration separately for IAM and
Bentham using validation CER, then WER (exact ties retain grid order). Baseline
remains a separate control. `validation_parameter_selections.csv` and metadata
record selected configurations, validation scores, source studies, and a frozen
configuration hash. `final_results_both_datasets.csv` presents the two test
corpora side by side. Interrupted superseded runs remain marked and separate;
matching baseline/grayscale checkpoints can be reused after identity checks.

Final Part 2 studies use `test/experiment_a_part_2/<UTC-run-id>/`. They restore
Part 1's exact test subsets and frozen dataset-specific validation parameters.
The six-stage order is recorded before inference; combined predictions are new,
while identical signed Part 1 baselines are reused with provenance recorded.
Outputs include two-dataset tables, Part 1 individual-versus-combined comparison,
paired CER plots, line outcomes, cumulative stage previews and diagnostic images.

Final Part 3 studies use `test/experiment_a_part_3/<UTC-run-id>/`. They compare
baseline, the exact Part 2 six-stage pipeline, and the same transforms with CLAHE
moved before denoising and deskew. Identity-checked Part 2 predictions are reused
with source paths recorded; only the reordered treatment is newly recognized.
Six dataset/configuration pairs cover the same 250 test lines per dataset.
Outputs include changes versus baseline and Part 2, paired diagnostics, previews,
two-dataset final tables, and a comparison with the Part 1 individual treatments.
