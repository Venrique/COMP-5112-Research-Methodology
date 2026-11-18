# Handwritten Text Recognition: Preprocessing Experiments

Research Methodology project investigating how image preprocessing affects
handwritten text recognition accuracy on IAM and Bentham text-line images.
The project includes pretrained Microsoft TrOCR, a CNN-BiLSTM-CTC recognizer
trained on IAM, configurable preprocessing, and a local Streamlit application.

The CNN training methodology and results are documented in
[docs/CNN_LSTM_CTC.md](docs/CNN_LSTM_CTC.md).

## Setup

The project environment uses Python 3.13. Run commands from the project root in
Windows PowerShell. If `.venv` does not exist, create it first:

```powershell
python -m venv .venv
```

Install the pinned dependencies into the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-frontend.txt
```

`requirements-frontend.txt` includes the inference and preprocessing dependencies.
For experiments without the app, install `requirements-inference.txt` instead.
Select this environment as the Python kernel in a Jupyter-capable editor.
Initial dataset and TrOCR downloads require access to Hugging Face.

`.venv/` contains installed dependencies, including `Lib/`, `Scripts/`, and
Jupyter settings under `etc/`. It is excluded from Git. Recreate it from the
requirements files on another machine.

## Run the app

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open the local address printed in the terminal, normally `http://localhost:8501`.

1. Choose **CNN + LSTM + CTC** or **TrOCR**.
2. Upload one cropped English handwritten text line as PNG, JPG, or WebP.
3. Select optional preprocessing and inspect the original and processed previews.
4. Optionally enter the correct text, then click **Read handwriting**.
5. View the prediction, inference time, and CER/WER when correct text is supplied.
6. Download the result as JSON, including model identity and preprocessing settings.

All preprocessing options start disabled, preserving the baseline. Enabled
options run in this fixed app order:

`grayscale -> CLAHE -> denoising -> deskew -> intensity normalization -> size normalization`

Expand **Preprocessing parameters** to inspect settings. App controls use
demonstration defaults, including resize height 64; these are separate from the
validation-selected experiment configurations. Each model still applies its
mandatory input preparation afterward.

Correct text is used only for scoring. Without it, error scores are marked
**Not calculated**. Editing it updates scores without another inference.
Changing the image, model, or preprocessing requires another recognition.
Uploads are processed in memory and do not modify the research datasets.

The CNN model uses the included checkpoint at
`results/cnn_lstm_ctc/training/20261114_baseline/iam/best.pt`.
TrOCR uses `microsoft/trocr-base-handwritten` with its pinned revision.
Both models run locally; TrOCR may need a first-use download. To override the
CNN checkpoint path before launching, set `$env:HTR_CNN_CHECKPOINT`.
See [docs/STREAMLIT_APP.md](docs/STREAMLIT_APP.md) for configuration and scoring.

## Project layout

| Path | Purpose |
| --- | --- |
| `src/load_dataset/` | IAM, Bentham, and local-image loaders |
| `src/preprocessing/` | Independent image transforms and ordered pipelines |
| `src/models/` | TrOCR and CNN-BiLSTM-CTC recognizers |
| `src/evaluation/` | Corpus CER/WER and configuration selection |
| `src/validation/` | Validation studies for each preprocessing technique |
| `src/training/` | CTC data preparation, training, checkpoints, and epoch reports |
| `src/frontend/` | App upload handling, model loading, and scoring |
| `notebooks/demo/` | Dataset, preprocessing, and recognition demonstrations |
| `notebooks/validation/` | Individual preprocessing parameter studies |
| `notebooks/experiments/` | Individual, combined, and reordered TrOCR comparisons |
| `notebooks/cnn_lstm_ctc/` | Preparation, model checks, training analysis, and local predictions |
| `results/` | Recorded predictions, configurations, tables, charts, and selected weights |
| `data/local/images/` | Local handwriting examples |
| `templates/local_images.csv` | Local-image paths and optional reference text |
| `tests/` | Dataset, preprocessing, metrics, model, training, and app-helper checks |

## Datasets

IAM is loaded from `Teklia/IAM-line`. Preserve its original train, validation,
and test splits. Use validation for parameter selection and test only for frozen
final comparisons. Record the dataset revision and source indices with each run.

Bentham is stored under the Git-ignored `data/bentham/` directory. To download
the pinned archives and verify their checksums:

```powershell
.\.venv\Scripts\python.exe -m src.load_dataset.bentham --download
```

Allow about 9 GB for download and extraction. The loader preserves the supplied
partitions, images, and transcriptions. Verify the dataset's usage terms before
redistributing it.

For local recognition, put one handwritten line per image in `data/local/images/`
and update `templates/local_images.csv`. Paths are relative to the CSV file.
Blank reference text permits unscored prediction. Run
[the TrOCR local-image notebook](notebooks/demo/05_local_images_recognition.ipynb)
or [the CNN local-image notebook](notebooks/cnn_lstm_ctc/04_local_images_recognition.ipynb).

## Preprocessing and experiments

The optional techniques are grayscale, non-local means denoising, deskewing,
CLAHE, global min-max intensity normalization, and aspect-preserving resizing.
Each can be enabled independently; pipeline order is explicit.

```python
from src.preprocessing import NonLocalMeans, PreprocessingPipeline

baseline = PreprocessingPipeline()
denoising = PreprocessingPipeline((NonLocalMeans(h=3.0, h_color=3.0),))
```

Run notebooks using the project environment, inspecting the configuration cells
before execution. [Validation notebooks](notebooks/validation/README.md) select
parameters independently for IAM and Bentham.

| Notebook | Comparison | Configured test samples |
| --- | --- | --- |
| [Experiment A](notebooks/experiments/experiment_a.ipynb) | Baseline and six individual techniques | 1,000 IAM and all 860 Bentham lines |
| [Part 2](notebooks/experiments/experiment_a_part_2.ipynb) | Baseline and combined preprocessing | Saved 250 IAM and 250 Bentham lines |
| [Part 3](notebooks/experiments/experiment_a_part_3.ipynb) | Baseline and two preprocessing orders | Same saved 250 lines per dataset as Part 2 |

Part 2 applies intensity normalization, grayscale, denoising, deskew, CLAHE, and
size normalization. Part 3 moves CLAHE before denoising and deskew while retaining
the other stages. These experiment orders differ from the app's display order.

Parts 2 and 3 restore the earlier 250-line Experiment A run, not the current
larger Part 1 configuration. Compare results only when dataset indices and
inference settings match. Notebook configuration describes the requested run;
saved metadata and completion status establish what was actually evaluated.

Compare treatments on identical ordered samples with fixed model weights and
decoding. Mandatory model input preparation follows optional preprocessing.
Scoring collapses whitespace and retains case and punctuation. Corpus metrics
aggregate edit counts, rather than averaging individual line rates; rates may
exceed 100%. Original images and transcriptions are preserved. Predictions,
metrics, model/dataset revisions, preprocessing settings, and software versions
are saved under `results/`; see [the results guide](results/README.md).

To preprocess a single image without recognition:

```powershell
.\.venv\Scripts\python.exe run_preprocessing.py input.png results/processed.png --steps clahe denoising deskew
```

## CNN baseline results

All saved CNN-BiLSTM-CTC figures and result CSVs are collected in
[results/cnn_bilstm_ctc_summary/](results/cnn_bilstm_ctc_summary/).
The [results index](results/cnn_bilstm_ctc_summary/README.md) explains each image,
its source notebook or plotting module, and the corresponding data files.
The collection includes 10 images, 34 result CSVs, and a file catalog with original
paths and checksums. Files prefixed `diagnostic_smoke_` show small implementation
checks, not the full IAM run.

IAM training stopped after epoch 17 with a maximum budget of 20 epochs.
Epoch 14 was selected by validation CER, then WER, then loss: **10.07% CER and
34.29% WER** on the fixed 100 validation lines. These are validation results;
CNN preprocessing comparisons and final test evaluation remain pending.

- [All epoch results](results/cnn_lstm_ctc/training/20261114_baseline/iam/EPOCH_RESULTS.md)
- [Training history](results/cnn_lstm_ctc/training/20261114_baseline/iam/history.csv)
- [Training curves](results/cnn_lstm_ctc/training/20261114_baseline/iam/training_curves.png)
- [Epoch 14 prediction examples](results/cnn_lstm_ctc/training/20261114_baseline/iam/examples/epoch_014_random_100_seed5112.md)
- [Notebook guide](notebooks/cnn_lstm_ctc/README.md)

The CNN learns from IAM training data locally, whereas TrOCR uses Microsoft's
pretrained weights. Report this difference when comparing architectures.
Opening the training notebook does not start training: `RUN_TRAINING=False`
allows inspection of the saved run.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Tests use small fixtures to check implementation behavior. They do not establish
recognition accuracy. Saved research results, model provenance, and evaluation
limitations are documented separately in `results/` and the training report.
Live browser behavior and real model inference through the app require manual
verification; unit tests cover the app's upload, preprocessing, and scoring helpers.
