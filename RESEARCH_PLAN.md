
# Research Plan: Handwritten Text Recognition

## Research topic

Evaluating the impact of image preprocessing techniques on the recognition accuracy of pretrained deep learning models for handwritten text recognition.

## Primary research question

How do different image preprocessing techniques affect the recognition accuracy of pretrained handwritten text recognition models?

## Secondary research questions

1. Which preprocessing techniques produce the greatest improvement in recognition accuracy?
2. Do Transformer-based HTR models benefit from preprocessing differently than other HTR architectures?
3. Do combinations of preprocessing techniques perform better than individual techniques?
4. Does preprocessing effectiveness vary across different handwriting datasets and languages?
5. How do handwriting characteristics and image quality influence recognition accuracy?

## Research motivation

Earlier HTR systems often relied on extensive image preprocessing pipelines.

Modern deep learning architectures may be more robust to input variations, potentially reducing the need for certain preprocessing operations.

The research aims to determine experimentally whether preprocessing improves accuracy, which techniques are beneficial, and whether their effects depend on model architecture or dataset characteristics.

## Planned HTR models

### TrOCR
- Transformer-based encoder-decoder architecture.
- Initial checkpoint: microsoft/trocr-base-handwritten.
- Primary starting point for implementation.

### HTR-VT
- Vision Transformer-based HTR architecture.
- Use a pretrained checkpoint if available and compatible.
- Verify the implementation and pretrained weights before integration.

### CNN + LSTM + CTC
- Add a CNN and bidirectional LSTM recognizer with CTC, trained from scratch as
  requested for the architecture comparison. Keep the pretrained TrOCR baseline.
- Current scope, revised by the user: train the CNN-LSTM-CTC baseline on IAM
  only, using its original training split. Use the teammate's fixed 100 IAM
  validation lines to select checkpoints and optional preprocessing settings;
  preserve the fixed 250 IAM test lines for final comparisons. Bentham preparation
  is retained, but full CNN-LSTM-CTC training on Bentham is deferred.
- Keep required grayscale/resize input preparation identical across a model's
  baseline and optional preprocessing treatments. Freeze the selected checkpoint
  before measuring preprocessing effects.
- The user extended the development run from a 4-epoch limit to a maximum of
  20 IAM epochs after reviewing early validation results. Keep the same
  validation-based early stopping and record this adaptive budget change.
  Document every completed epoch's loss, corpus CER/WER, fixed prediction
  examples, and available checkpoint. Assess convergence on validation rather
  than assuming this budget is sufficient; reserve test for final evaluation.
- Report the difference in training provenance: CNN-LSTM-CTC learns locally from
  dataset training splits, whereas TrOCR uses Microsoft's existing pretrained weights.
- Completed IAM baseline: early-stopped after 17 epochs; epoch 14 selected by
  lowest validation CER, then WER, then validation loss. Selected validation
  CER is 10.07% and WER 34.29% on the fixed 100 IAM lines. Epoch 13 has the
  lowest WER, so the selected checkpoint is not best on every metric.
- See `docs/CNN_LSTM_CTC.md` for complete methodology, all 17 epoch results,
  verified actual/predicted text examples, checkpoint provenance, and limitations.
  CNN-LSTM-CTC preprocessing experiments and final test evaluation remain pending.

TrOCR is used for inference and evaluation without local fine-tuning. The new
CNN-LSTM-CTC comparison includes the explicitly requested from-scratch training.

## Datasets

### IAM — English
Source: https://huggingface.co/datasets/Teklia/IAM-line

Use line-level images and ground-truth text.

The dataset loader has already been implemented and tested.

The test split was successfully loaded with 2,915 samples.

The dataset provides `image` and `text` columns.

### RIMES — French
Potential second dataset for language generalization experiments.

### Other languages
Consider additional handwritten datasets, including Spanish, if compatible pretrained models and appropriate evaluation data are available.

## Preprocessing techniques

Implement the following techniques as independent, configurable operations:

1. Grayscale conversion
2. Noise removal / denoising
3. Deskewing
4. Contrast enhancement

Potential future techniques:
- Binarization
- Resizing experiments
- Morphological operations

Distinguish optional experimental preprocessing from mandatory model-specific input processing.

## Experimental design

### Experiment A — Individual preprocessing techniques

Hold the model and evaluation dataset constant.

Compare:
- Original image
- Grayscale
- Denoising
- Deskewing
- Contrast enhancement

Identify the individual effect of each technique.

### Experiment B — Combined preprocessing

Evaluate selected combinations of techniques against the baseline.

Make preprocessing order explicit.

### Experiment C — Model comparison

Evaluate TrOCR and HTR-VT on the same datasets and samples.

Document differences in training data, vocabulary, and model capacity.

Evaluate how preprocessing effects vary between model architectures.

### Experiment D — Dataset and language comparison

Repeat selected experiments on another language dataset, such as RIMES.

Compare improvements relative to each dataset's baseline.

Do not attribute absolute accuracy differences solely to language, since pretrained models may have different language and training-data coverage.

## Evaluation metrics

### Character Error Rate (CER)
Character-level edit distance divided by the number of reference characters.

### Word Error Rate (WER)
Word-level edit distance divided by the number of reference words.

Lower values are better.

Spaces and punctuation can affect CER/WER.

Use a consistent, documented text-normalization policy.

Report dataset-level metrics and consider per-sample error distributions.

### Additional metrics
- Inference time
- Relative and absolute changes in CER/WER
- Number of evaluated samples
- Optional uncertainty estimates

## Research reproducibility

Each experiment should record:
- Experiment ID
- Dataset name and split
- Dataset revision when available
- Model name and checkpoint
- Preprocessing configuration and execution order
- CER and WER
- Inference time
- Number of samples
- Relevant software versions
- Random seed, if applicable

Save results in CSV or another structured format.

Never mix validation and final test results without identifying them.

## Expected outputs

An optional local Streamlit diagnostic interface is implemented. Users upload
one handwritten line, select the existing CNN-LSTM-CTC or TrOCR recognizer,
and receive a prediction. Per-image CER/WER require user-supplied ground truth
and use the same scoring policy as the experiment code. App uploads do not
modify datasets or enter the reserved test experiments. Optional preprocessing
is selectable with independent checkboxes and recorded in downloads; all off
preserves the baseline. No app training is performed. Launch and real frontend inference are
left to the user; see `docs/STREAMLIT_APP.md`.

1. Reusable Python modules
2. Configurable experimental pipeline
3. Jupyter Notebooks demonstrating experiments
4. Recognition examples
5. Results tables and comparison charts
6. Research discussion and conclusions

## Current implementation status

Completed:
- Initial IAM Hugging Face dataset loading
- Ground-truth extraction
- Display of handwritten images and transcriptions
- Configurable optional preprocessing with CLAHE, non-local means denoising, and deskewing
- TrOCR inference and corpus CER/WER evaluation
- CNN-LSTM-CTC image/label preparation, architecture and decoder, training,
  checkpoints/resume, and validation-based early stopping
- IAM-only baseline training through epoch 17, with epoch 14 selected
- Per-epoch tables, loss/error graphs, actual predictions, checkpoint archives,
  and a verified 100-example random-order export from the same validation set

Next:
- Freeze the selected CNN-LSTM-CTC epoch 14 checkpoint and choose optional
  preprocessing settings/combined-order treatments on validation
- Run matched TrOCR/CNN-LSTM-CTC final comparisons on the saved 250 IAM test
  indices only after the validation protocol is fixed
- Verify pretrained HTR-VT checkpoint availability, license, inputs, and decoding before integration
- Reserve the IAM test split for final evaluation
