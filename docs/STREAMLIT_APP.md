# Streamlit handwriting frontend

## Run the app

From the project root in Windows PowerShell, using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-frontend.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Then open the local address printed by Streamlit, normally
`http://localhost:8501`. Stop the app with Ctrl+C in that terminal. The server
binds to loopback by default.

`requirements-frontend.txt` includes the existing inference requirements and
pins Streamlit 1.67.0. Live browser behavior and real frontend inference remain
to be verified manually.

### If you see an unexpected-keyword error

Stop the server with Ctrl+C and use the explicit `.venv` commands above to
ensure the app runs with the pinned project dependencies. A bare `streamlit run`
may use a different global installation.

## User flow

### Optional preprocessing controls

After uploading, enable independent checkboxes beneath the original preview.
All options start off (baseline). A second preview shows the exact processed
image passed to the recognizer before its mandatory preparation. The fixed order
is grayscale → CLAHE → denoising → deskew → intensity normalization → size normalization;
only checked steps run. Expand **Preprocessing parameters** to inspect settings.
These use the existing modules' defaults, not validation-selected optima:
CLAHE clip 2/grid 8×8; denoising strengths 3, patch 7/search 21; deskew uses
the existing bounded angle search; global min-max intensity normalization;
aspect-preserving LANCZOS resize to height 64.

Changing options clears the old prediction while retaining the same image's
reference text. Click **Read handwriting** again to recognize the processed
image. Downloads include ordered enabled steps and parameters. Optional
processing time is excluded from reading time. Mandatory model preparation
always remains enabled, including CNN grayscale/height normalization.

1. Choose **CNN + LSTM + CTC** or **TrOCR** in the sidebar.
2. Upload a single cropped English handwritten text line (PNG, JPG or WebP).
3. Optionally enter the exact correct text, preserving punctuation/case.
4. Click **Read handwriting**. Only that selected model loads and predicts.
5. Read the predicted text and inference time. If correct text is supplied,
   the app also shows this image's CER and WER.
6. Optionally download a JSON result containing raw text, scores, image hash,
   model configuration, checkpoint identity, and inference time.

Correct text is never passed to the recognizer. Without it, scores say
**Not calculated**, never zero or invented values. You can edit the correct
text after prediction; metrics update without another model inference.
Changing the model preserves the reference for the same image, but hides any
old prediction until the selected model is run. A new image clears the old
reference and prediction to prevent scoring against another image's text.

Empty model output is shown explicitly. It still receives valid error scores
when a nonempty reference is provided. Invalid, unsupported, animated, over-10MB,
or over-20-megapixel images receive a readable error before model loading.
The app does not detect/crop full pages into text lines.

## Model requirements

### CNN + LSTM + CTC

The default checkpoint is the trained IAM **epoch 14** model at:

```text
results/cnn_lstm_ctc/training/20261114_baseline/iam/best.pt
```

The selected checkpoint is included in Git at the default path; no separate
weights download is needed. Other training checkpoints remain Git-ignored. The app
displays an actionable missing-checkpoint message and disables CNN inference
if the selected file is absent.

To use another trusted project checkpoint, set its location before launch:

```powershell
$env:HTR_CNN_CHECKPOINT = 'C:\path\to\best.pt'
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

The existing `CNNLSTMCTCRecognizer.from_checkpoint` restores weights, character
IDs, mandatory height/grayscale preparation and checkpoint provenance. The
actual loaded epoch is shown in prediction details and recorded in downloads.

### TrOCR

The app uses the existing `TrOCRRecognizer` with:

- Checkpoint: `microsoft/trocr-base-handwritten`.
- Pinned revision: `eaacaf452b06415df8f10bb6fad3a4c11e609406`.
- CPU and deterministic generation: 128 maximum new tokens, one beam, no sampling.

The wrapper resolves the pinned model through Hugging Face, so internet access
may be needed to load it even when some weights are cached. A first-use download
can take time and disk space. The app does not send uploaded handwriting to an
inference API; both models perform inference on the local Python server.

Only the selected model loads after the button click. Streamlit caches up to
two model resources; replacing the CNN checkpoint changes the cache identity.
A lock serializes access to each shared cached model across sessions. Uploaded
images/results are not stored in the global model cache.

## Image preparation and scoring

Uploads are decoded into new in-memory images, corrected for EXIF orientation,
composited onto white, and converted to RGB. Original files are not overwritten
and uploads are not written to the dataset or results folders. Each model
then applies selected optional preprocessing followed by its own mandatory input
preparation. No retraining or spelling correction runs in the app.

CER/WER reuse `src/evaluation/recognition.py` with one reference/prediction
pair. Whitespace is collapsed; case and punctuation are retained. Lower is
better, and rates can exceed 100% because extra characters/words count as edits.
The UI shows percentages; downloaded JSON preserves fractional values. These
are per-upload diagnostics, not validation aggregates or final test results.
Correct-label errors also affect scores; the app does not guess ground truth.

Reading time includes required image preparation inside the recognizer but
excludes loading/caching the model and upload decoding. User data remains in
the running Streamlit session's memory; saving a result requires the download
button. No existing research records or checkpoints are modified.

## Files and checks

| File | Purpose |
| --- | --- |
| `streamlit_app.py` | Interface, session state, selected-model cache, prediction button, scores and JSON download |
| `src/frontend/recognition.py` | Upload decoding, checkpoint/input identity, locked model resources and existing metric integration |
| `src/frontend/__init__.py` | Frontend helper package |
| `.streamlit/config.toml` | Light theme, local address, upload limit, usage telemetry disabled |
| `requirements-frontend.txt` | Existing inference requirements plus pinned Streamlit |
| `tests/test_frontend_recognition.py` | Offline helper tests using fake predictors/factories; no Streamlit launch or model downloads |

Checks cover upload validation, alpha/EXIF handling, source-image preservation,
missing reference behavior, empty predictions, rates above 100%, scoring policy,
model/input cache identity, missing checkpoint messaging, and selected-model
factory isolation. Syntax checks compile the entry file without executing it.
The browser, dependency installation, and real model inference through this
frontend are deliberately untested until manual launch. Existing notebook-based
inference results are separate from frontend verification.

Next: manually install/launch, try both models with one known local image and
its correct text, and verify the layout and downloaded result. Avoid tuning
using reserved IAM test examples.

References used for implementation: [file uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader),
[resource caching](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource),
[download button](https://docs.streamlit.io/develop/api-reference/widgets/st.download_button),
[pinned dependency metadata](https://pypi.org/pypi/streamlit/1.67.0/json).
