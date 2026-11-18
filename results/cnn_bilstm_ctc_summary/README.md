# CNN-BiLSTM-CTC results index

This collection contains **10 images and 34 original result CSVs**, copied without changing their contents. Original run folders remain under `results/cnn_lstm_ctc/`. The additional [file_catalog.csv](file_catalog.csv) maps every copied file to its original location, related notebook, source code, data source, and SHA-256 hash.

## Start here

- [Training loss and validation CER/WER graph](images/iam_training_loss_and_validation_errors.png)
- [Training history: all 17 epochs](csv/training/20261114_baseline/iam/history.csv)
- [Detailed epoch summary](csv/training/20261114_baseline/iam/epoch_summary.csv)
- [Selected epoch 14: ground truth and predictions](csv/training/20261114_baseline/iam/validation_epoch_014.csv)
- [Local handwriting images and predictions](images/local_images_and_predictions.png)
- [Local prediction details](csv/local_predictions/20261115T054718447180Z/predictions.csv) and [aggregate local scores](csv/local_predictions/20261115T054718447180Z/summary.csv)

The full IAM run trained on 6,482 lines and validated on the same 100 lines each epoch. It stopped after 17 epochs; epoch 14 was selected with **10.07% validation CER and 34.29% validation WER**. These are validation scores. CNN final test evaluation and optional preprocessing comparisons remain pending.

## Image-to-source mapping

Notebook cell numbers below count all cells from 1. Source code links explain where the figures are assembled or plotted; display cells simply show saved images.

| Image | What it shows | Notebook / plotting code | Underlying data |
| --- | --- | --- | --- |
| [bentham_original_and_prepared.png](images/bentham_original_and_prepared.png) | Original line beside mandatory grayscale/height-64 preparation | [01_data_preparation.ipynb](../../notebooks/cnn_lstm_ctc/01_data_preparation.ipynb); cell 8; exported in 18 | [metadata.json](../../results/cnn_lstm_ctc/data_preparation/20261114T211708884735Z/metadata.json) |
| [bentham_word_to_ids.png](images/bentham_word_to_ids.png) | Ground-truth character encoding, not a model prediction | [01_data_preparation.ipynb](../../notebooks/cnn_lstm_ctc/01_data_preparation.ipynb); cell 10; exported in 18 | [metadata.json](../../results/cnn_lstm_ctc/data_preparation/20261114T211708884735Z/metadata.json) |
| [dataset_split_counts.png](images/dataset_split_counts.png) | Predefined dataset split sizes and fixed comparison subsets | [01_data_preparation.ipynb](../../notebooks/cnn_lstm_ctc/01_data_preparation.ipynb); cell 12; exported in 18 | [metadata.json](../../results/cnn_lstm_ctc/data_preparation/20261114T211708884735Z/metadata.json) |
| [iam_original_and_prepared.png](images/iam_original_and_prepared.png) | Original line beside mandatory grayscale/height-64 preparation | [01_data_preparation.ipynb](../../notebooks/cnn_lstm_ctc/01_data_preparation.ipynb); cell 8; exported in 18 | [metadata.json](../../results/cnn_lstm_ctc/data_preparation/20261114T211708884735Z/metadata.json) |
| [iam_word_to_ids.png](images/iam_word_to_ids.png) | Ground-truth character encoding, not a model prediction | [01_data_preparation.ipynb](../../notebooks/cnn_lstm_ctc/01_data_preparation.ipynb); cell 10; exported in 18 | [metadata.json](../../results/cnn_lstm_ctc/data_preparation/20261114T211708884735Z/metadata.json) |
| [training_text_lengths.png](images/training_text_lengths.png) | Training-transcription length distributions | [01_data_preparation.ipynb](../../notebooks/cnn_lstm_ctc/01_data_preparation.ipynb); cell 14; exported in 18 | [metadata.json](../../results/cnn_lstm_ctc/data_preparation/20261114T211708884735Z/metadata.json) |
| [local_images_and_predictions.png](images/local_images_and_predictions.png) | Ten local handwriting images with actual and predicted text from IAM epoch 14 | [04_local_images_recognition.ipynb](../../notebooks/cnn_lstm_ctc/04_local_images_recognition.ipynb); cell 8 (plot); 10 (export) <br> [src/evaluation/recognition.py](../../src/evaluation/recognition.py); compare_recognition; figure composed in notebook | [predictions.csv](../../results/cnn_lstm_ctc/local_predictions/20261115T054718447180Z/predictions.csv) <br> [summary.csv](../../results/cnn_lstm_ctc/local_predictions/20261115T054718447180Z/summary.csv) <br> [metadata.json](../../results/cnn_lstm_ctc/local_predictions/20261115T054718447180Z/metadata.json) |
| [iam_training_loss_and_validation_errors.png](images/iam_training_loss_and_validation_errors.png) | Training/validation CTC loss and validation CER/WER over 17 epochs | [03_training.ipynb](../../notebooks/cnn_lstm_ctc/03_training.ipynb); cell 6 (displays saved plot) <br> [src/training/ctc_training.py](../../src/training/ctc_training.py); plot_history(run_dir) | [history.csv](../../results/cnn_lstm_ctc/training/20261114_baseline/iam/history.csv) |
| [diagnostic_smoke_bentham_training_curves.png](images/diagnostic_smoke_bentham_training_curves.png) | One-epoch smoke-test loss and validation errors; not full training | [src/training/ctc_training.py](../../src/training/ctc_training.py); plot_history(run_dir) | [history.csv](../../results/cnn_lstm_ctc/training_smoke/20261114_first_run/bentham/history.csv) |
| [diagnostic_smoke_iam_training_curves.png](images/diagnostic_smoke_iam_training_curves.png) | One-epoch smoke-test loss and validation errors; not full training | [src/training/ctc_training.py](../../src/training/ctc_training.py); plot_history(run_dir) | [history.csv](../../results/cnn_lstm_ctc/training_smoke/20261114_first_run/iam/history.csv) |

## CSV folders and meaning

| Location | Contents and source |
| --- | --- |
| [csv/training/20261114_baseline/iam/](csv/training/20261114_baseline/iam/) | Full-run `history.csv`, `epoch_summary.csv`, validation predictions for epochs 001-017, and train/validation input audits. Training exports: [ctc_training.py](../../src/training/ctc_training.py); verified summary: [epoch_reporting.py](../../src/training/epoch_reporting.py). |
| [csv/training/20261114_baseline/iam/examples/](csv/training/20261114_baseline/iam/examples/) | All 100 selected-epoch validation examples in shuffled order. Derived from `validation_epoch_014.csv`; seed 5112 and the exact procedure are recorded in the [original manifest](../../results/cnn_lstm_ctc/training/20261114_baseline/iam/examples/epoch_014_random_100_seed5112.json). |
| [csv/local_predictions/20261115T054718447180Z/](csv/local_predictions/20261115T054718447180Z/) | Predictions for 10 local images and their corpus summary. Produced by notebook 04, cell 10, using the shared evaluator. |
| [csv/input_audit/](csv/input_audit/) | Bentham preflight alignment checks: 9,198 training lines and 100 validation lines. These are input checks, not full Bentham training results. The matching helper is `audit_alignment`; a separate export launcher is not retained. |
| [csv/training_smoke/](csv/training_smoke/) | Separate one-epoch IAM and Bentham implementation checks on 16 training and 16 validation examples each. These are not the full IAM results. |

## Reading and explaining the files

- In the training graph, the left panel plots CTC loss; the right panel plots validation CER and WER as percentages. All four curves come from the matching `history.csv`.
- CSV `cer`, `wer`, `validation_cer`, and `validation_wer` fields are fractions; multiply by 100 for percentages. Fields ending `_pct` already contain percentages. Lower is better.
- `validation_epoch_014.csv` contains the selected checkpoint's saved predictions. The shuffled example export contains the same validation lines, not additional evaluation.
- The local recognition image is backed by the local `predictions.csv`. It uses the IAM epoch-14 model on the local manifest, not the IAM test set.
- `word_to_ids` images illustrate ground-truth character encoding, not model predictions. Bentham preparation figures do not imply completed full Bentham training.
- Original paths recorded inside copied CSVs keep their original meaning. For example, checkpoint references in `epoch_summary.csv` resolve against the [original IAM run](../../results/cnn_lstm_ctc/training/20261114_baseline/iam/). Model weights are not duplicated here.
- This index describes existing saved outputs. It does not rerun training, recognition, or evaluation.
