
# Research Plan: Handwritten Text Recognition

## Research topic

Evaluating the impact of image preprocessing techniques on the recognition accuracy of pretrained deep learning models for handwritten text recognition.

## Primary research question

How do different image preprocessing techniques affect the recognition accuracy of pretrained handwritten text recognition models?

## Secondary research questions

1. Which preprocessing techniques produce the greatest improvement in recognition accuracy?
2. Do modern Transformer-based HTR models benefit from preprocessing as much as traditional CNN-RNN architectures?
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
- CNN extracts image features.
- LSTM processes sequential relationships.
- CTC handles sequence alignment and decoding.
- Find a suitable pretrained handwriting recognition implementation.

The goal is inference and evaluation, not training from scratch.

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

Evaluate TrOCR, HTR-VT, and CNN + LSTM + CTC on the same datasets and samples.

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

Next:
- Integrate pretrained TrOCR inference
- Implement CER and WER
- Implement configurable preprocessing
