
# HTR Research Project — Codex Instructions

## Project overview

This is an academic research project for a Research Methodology course.

The project investigates how image preprocessing techniques affect the recognition accuracy of pretrained Handwritten Text Recognition (HTR) models.

The primary goal is to conduct controlled, reproducible experiments — not to develop a new neural network architecture or train models from scratch.

Read `RESEARCH_PLAN.md` for the detailed research methodology.

## Core objectives

1. Load handwriting datasets with ground-truth transcriptions.
2. Implement configurable image preprocessing techniques.
3. Integrate pretrained HTR models.
4. Compare recognition accuracy with and without preprocessing.
5. Evaluate models using Character Error Rate (CER) and Word Error Rate (WER).
6. Investigate whether preprocessing effectiveness varies across model architectures and datasets.
7. Generate tables, charts, and examples for an academic research presentation.

## Technology stack

- Python
- Jupyter Notebooks for experiments and visualization
- PyTorch and Hugging Face Transformers for compatible models
- Hugging Face Datasets for dataset loading
- OpenCV and Pillow for image processing
- Pandas and NumPy for results analysis
- Matplotlib for visualization
- JiWER or an equivalent validated implementation for CER/WER

Use the existing Python environment and dependency management conventions.

## Development architecture

Keep reusable logic in Python modules under `src/`.

Use Jupyter Notebooks under `notebooks/` for:
- Dataset exploration
- Experiment configuration
- Running comparisons
- Visualizing images and predictions
- Analyzing CER/WER results

Keep notebooks lightweight. Do not duplicate implementation logic across notebooks.

Design model wrappers with a consistent interface such as:

    prediction = model.predict(image)

Design preprocessing functions so that each technique can be enabled or disabled independently.

Design experiments so that model choice, dataset split, preprocessing configuration, and evaluation parameters can be changed without rewriting code.

## Model architectures

The planned models are:

1. TrOCR — Microsoft's pretrained Transformer-based handwriting recognition model.
2. HTR-VT — Vision Transformer-based handwriting recognition.
3. CNN + LSTM + CTC — A conventional convolutional recurrent architecture.

Start with pretrained TrOCR.

Before integrating other models, verify checkpoint availability, licenses, input requirements, output decoding, and dependencies.

Do not train or fine-tune models unless explicitly requested.

## Dataset handling

The initial dataset is `Teklia/IAM-line` from Hugging Face.

It contains handwritten text-line images and ground-truth transcriptions.

The existing loader is in `src/datasets/iam.py`.

Preserve the dataset's predefined train, validation, and test splits.

Use validation data for development and preprocessing selection. Reserve the test set for final evaluation.

Possible future datasets include RIMES (French) and multilingual handwriting datasets.

## Experimental integrity

This is a scientific research project. Reproducibility and fair comparisons are essential.

- Never fabricate results, predictions, or performance metrics.
- Preserve original images and original ground-truth transcriptions.
- Establish a baseline without optional preprocessing.
- Evaluate preprocessing techniques individually before testing combinations.
- Compare configurations on the same evaluation samples.
- Keep model checkpoints and inference settings fixed when testing preprocessing effects.
- Apply each model's mandatory input transformations consistently.
- Document differences in model training data and supported languages.
- Avoid test-set leakage and repeated tuning on test results.
- Report aggregate CER/WER over the evaluation dataset, rather than relying only on individual examples.
- Record experimental configurations, dataset revisions, checkpoint names, and library versions.
- Keep text normalization consistent and explicitly documented.

## Collaboration and coding conventions

Two students are working on this project.

Write modular, readable, documented Python code.

Use type hints where helpful and docstrings for reusable functions.

Add small tests for preprocessing, dataset loading, and metric calculations.

Prefer simple implementations over unnecessary abstractions.

Before making major structural changes, explain the proposed changes.

Do not delete existing work, datasets, or experiment results without explicit approval.

## Codex workflow

Before starting a substantial task:
1. Inspect relevant project files.
2. Read `RESEARCH_PLAN.md`.
3. Summarize the intended implementation.
4. Identify dependencies and any risks.

After implementation:
1. Run relevant checks or tests where possible.
2. Explain the files changed.
3. Report what was verified and what remains untested.
4. Suggest the next logical development task.

Prioritize correctness, reproducibility, and clarity over complexity.
