# CNN + LSTM + CTC notebook guide

## Final result

The full IAM run is **finished through early stopping at epoch 17**, with a
requested maximum of 20. The selected checkpoint is **epoch 14**, validation
CER **10.07%** and WER **34.29%** on the same 100 validation lines. No CNN-LSTM-CTC
test evaluation or optional preprocessing experiment has run yet.

- [Complete implementation and research report](../../docs/CNN_LSTM_CTC.md)
- [Short final run summary](../../results/cnn_lstm_ctc/training/20261114_baseline/iam/FINAL_SUMMARY.md)
- [All completed epoch results](../../results/cnn_lstm_ctc/training/20261114_baseline/iam/EPOCH_RESULTS.md)
- [100 epoch 14 examples](../../results/cnn_lstm_ctc/training/20261114_baseline/iam/examples/epoch_014_random_100_seed5112.md)

## Open the notebooks in order

Use the existing project Python environment as the kernel and run code from
that environment. No additional dependencies were introduced.

| Notebook | What it shows |
| --- | --- |
| [01_data_preparation.ipynb](01_data_preparation.ipynb) | Original/prepared images, labels as numbers, train/validation/test sizes, training text-length graphs, batches |
| [02_model_check.ipynb](02_model_check.ipynb) | Forward output, CTC loss, gradients without optimizer updates, synthetic `hello` decoder example |
| [03_training.ipynb](03_training.ipynb) | Final training status, 17-epoch history, loss/error graphs, epoch documentation, checkpoint loading |
| [04_local_images_recognition.ipynb](04_local_images_recognition.ipynb) | Selected epoch 14 reads local images, with previews, predictions and diagnostic scores |

Opening the third notebook does not start training; leave `RUN_TRAINING=False`
to inspect saved results. Its report/status/graph cells can be rerun safely.
The first two notebooks include Bentham because both datasets were initially
prepared and checked. Full research training was later restricted to IAM.

## Data and mandatory preparation

- Preserve the original splits: IAM training 6,482, validation 976, test 2,915.
- Use all training lines and the teammate's exact saved 100 validation indices.
  The saved 250 IAM test indices remain reserved for final comparisons.
- Build the 79-character vocabulary from training text only. ID 0 is CTC blank.
  Unknown characters are reported/rejected; never extend the vocabulary using
  validation/test text.
- Retain raw ground truth. Labels/scoring collapse whitespace and retain case
  and punctuation.
- Composite transparency onto white, convert to grayscale, resize to height 64
  proportionally with LANCZOS, and map pixels to [0,1]. Originals are preserved.
- Right-pad batch images with white and concatenate unpadded CTC label IDs.
  Preserve pixel widths and label lengths separately.
- The CNN outputs floor(width/4) time positions. Repeated adjacent letters need
  extra blank positions; insufficient lengths fail explicitly rather than
  silently removing examples or zeroing losses.
- Mandatory grayscale/resize apply to baseline and all future treatments.
  Optional grayscale may be redundant; experimental resizing still precedes
  mandatory resizing. No optional preprocessing ran during this baseline.

## Model, training, and results

Three CNN blocks (32/64/128 channels) feed a two-layer bidirectional LSTM
(hidden size 128), character classifier and CTC. The IAM model has 772,688
parameters and 80 output classes. Padding is excluded before CNN processing
and by packed LSTM sequences. Greedy CTC decoding has no language model or
spelling correction. Real implementation lives in `src/`, not these notebooks.

Training used Adam 0.001, batch 4, seed 5112, clip norm 5 and four CPU threads.
Validation checkpoints are ranked by CER, then WER, then loss; patience is 3.
Epoch 14 has best CER, epoch 13 best WER (34.19%), and epoch 16 lowest validation
loss. Epochs 15-17 did not beat the selected score, so epochs 18-20 do not exist.
The full results table and limitations are in the linked research report.

The final run folder is `results/cnn_lstm_ctc/training/20261114_baseline/iam/`:

| File | Meaning |
| --- | --- |
| `best.pt` | Selected epoch 14 for inference/comparisons |
| `last.pt` | Completed epoch 17, optimizer, history and RNG/shuffle states |
| `epoch_checkpoints/` | Available snapshots for epochs 2-17; no epoch 1 weights remain |
| `history.csv` | Original full-precision training/validation metrics and elapsed times |
| `training_curves.png` | Completed-epoch training/validation loss and validation CER/WER |
| `validation_epoch_NNN.csv` | Correct text and actual model output on the same 100 lines |
| `EPOCH_RESULTS.md` / `epoch_summary.csv` | Verified per-epoch explanation and analysis table |
| `FINAL_SUMMARY.md` / `FINAL_VERIFICATION.json` | Final status/results and artifact checks |
| `examples/epoch_014_random_100_seed5112.*` | Table, CSV and manifest for all 100 validation lines in random order |
| `metadata.json` / `status.json` / `progress.json` | Original configuration, final stopping state, last batch progress |
| `train_input_audit.csv` / `validation_input_audit.csv` | Header/label alignment checks before training |

`progress.json` shows epoch 17's last batch after completion; consult
`status.json` to determine whether a run is active. `.pt` files are binary,
not text, and ignored by Git. Preserve them separately with recorded hashes.
The random 100 examples are the same validation set shuffled with seed 5112,
not new evaluation data. Do not use `training_smoke/` diagnostic results as the
full model's results.

## Checkpoint loading and resume

```python
from src.models import CNNLSTMCTCRecognizer

recognizer = CNNLSTMCTCRecognizer.from_checkpoint(
    ROOT / "results/cnn_lstm_ctc/training/20261114_baseline/iam/best.pt",
    device="cpu",
)
# prediction = recognizer.predict(a_pillow_line_image)
```

Loading restores vocabulary/preparation and records checkpoint hash/epoch.
Epoch-boundary resume restores optimizer and shuffle/RNG states and repeats
only unfinished partial epochs. Identity checks reject changed model/data/
software/training settings except for the epoch limit.

This run has already early-stopped with patience counter 3. Increasing only
`--epochs` does not restart training. Further continuation requires a separately
documented policy/run that preserves the parent checkpoint. No new training
or inference was started during final documentation.

The root README and full report document the scope/budget changes, supervisor
artifacts, all source/test files, verification, and next validation-only
preprocessing experiments. Preserve test data for the final fixed comparison.

## Local handwriting diagnostic

Notebook 04 reads `templates/local_images.csv`, checks that the selected
manifest covers all images under `data/local/`, and loads `best.pt` (epoch 14).
The existing loader handles EXIF orientation and white alpha compositing before
the model's mandatory preparation. Only the empty optional pipeline runs.
Blank labels permit unscored prediction; filled labels are preserved as supplied.

The check covered 10 labeled images: **28.10% CER and 78.67% WER**, reported
separately from IAM validation/test. It saved predictions, corpus summary,
source hashes, manifest snapshot, checkpoint hash/epoch, software versions,
a readable table, and an image preview under
`results/cnn_lstm_ctc/local_predictions/20261115T054718447180Z/`.
Rerunning creates a new folder. No training or optional preprocessing occurred;
source image, manifest, and checkpoint bytes were checked unchanged.
