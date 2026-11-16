r"""Manual launch: .venv\Scripts\python.exe -m streamlit run streamlit_app.py"""
import hashlib
import json
import os
from pathlib import Path

import streamlit as st

from src.frontend.recognition import (
    DEFAULT_CNN_CHECKPOINT, MODEL_LABELS, PREPROCESSING_LABELS,
    build_preprocessing, checkpoint_token, decode_upload,
    load_resource, prediction_key, score_prediction,
)


@st.cache_resource(max_entries=2, show_spinner=False)
def get_model(model_key: str, checkpoint_path: str, token: str):
    # token is part of the cache key so replacing local weights invalidates cache.
    return load_resource(model_key, checkpoint_path)


def main() -> None:
    st.set_page_config(page_title="Handwriting Lab", page_icon="✍️", layout="wide")
    st.markdown("""
    <style>
    .block-container {max-width: 1240px; padding-top: 2.5rem; padding-bottom: 2rem;}
    [data-testid="stSidebar"] {border-right: 1px solid #E2E8F0;}
    .brand {font-size: 1.2rem; font-weight: 750; letter-spacing: -.035em; color: #0F172A;}
    .brand-mark {display:inline-flex; align-items:center; justify-content:center;
        width:34px; height:34px; border-radius:10px; margin-right:8px;
        color:white; background:#2563EB; font-size:20px;}
    .eyebrow {font-size: .73rem; letter-spacing:.15em; color:#2563EB; font-weight:700; margin-bottom:10px;}
    .hero-title {font-size:clamp(2rem,4vw,3rem); line-height:1.1;
        letter-spacing:-.05em; font-weight:750; color:#0F172A; margin:0 0 12px;}
    .hero-description {color:#64748B; font-size:1.02rem; margin-bottom:1.8rem; max-width:650px;}
    .empty-preview {min-height:220px; border:1px dashed #CBD5E1; border-radius:14px;
        background:#F8FAFC; display:flex; flex-direction:column; align-items:center;
        justify-content:center; color:#64748B; text-align:center; padding:25px;}
    .empty-preview strong {font-size:1rem; color:#334155; margin-bottom:8px;}
    [data-testid="stMetric"] {background:#FFFFFF; border:1px solid #E2E8F0;
        border-radius:14px; padding:18px 20px;}
    [data-testid="stMetricValue"] {font-size:1.7rem;}
    div.stButton > button {width:100%; border-radius:10px; font-weight:600; min-height:46px;}
    .footer-note {font-size:.78rem; color:#94A3B8; margin-top:2rem;}
    </style>
    """, unsafe_allow_html=True)

    checkpoint = Path(os.environ.get("HTR_CNN_CHECKPOINT", str(DEFAULT_CNN_CHECKPOINT))).expanduser().resolve()
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-mark">H</span>Handwriting Lab</div>', unsafe_allow_html=True)
        st.caption("HTR research project")
        st.divider()
        st.markdown("**Choose a recognizer**")
        selected = st.radio("Recognition model", list(MODEL_LABELS),
                            format_func=MODEL_LABELS.get, label_visibility="collapsed")
        if selected == "cnn_lstm_ctc":
            st.caption("Your CNN and bidirectional LSTM, trained on IAM and decoded with CTC.")
        else:
            st.caption("Microsoft's pretrained Transformer handwriting recognizer.")
            st.info("The first TrOCR prediction may download model weights. Later requests reuse the loaded model.")
        st.divider()
        st.markdown("**For a useful result**")
        st.caption("Upload one cropped line of English handwriting. Keep the text upright and clearly visible.")
        st.caption("Full-page text detection and line cropping are not included.")
        st.divider()
        st.caption("Local inference · CPU · No training")

    st.markdown('<div class="eyebrow">HANDWRITING TO TEXT</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Read the words.<br>Measure the mistakes.</h1>', unsafe_allow_html=True)
    st.markdown('<div class="hero-description">Upload a handwritten line, choose a model, and see its prediction. Add the correct text to measure character and word errors.</div>', unsafe_allow_html=True)

    token = checkpoint_token(selected, checkpoint)
    missing_checkpoint = selected == "cnn_lstm_ctc" and token.startswith("missing:")
    if missing_checkpoint:
        st.error("The trained CNN-LSTM-CTC checkpoint is missing. Restore best.pt on this computer before choosing this model.")
        with st.expander("Checkpoint location"):
            st.code(str(checkpoint), language=None)
            st.caption("You can also set HTR_CNN_CHECKPOINT to the saved checkpoint's path, then restart the app.")

    left, right = st.columns([1.05, 1], gap="large")
    image, processed_image, image_data, current_key = None, None, None, None
    with left:
        with st.container(border=True):
            st.markdown("### 1. Your handwriting")
            st.caption("One image · PNG, JPG or WebP · up to 10 MB")
            upload = st.file_uploader("Upload a handwriting image", type=["png", "jpg", "jpeg", "webp"],
                                      label_visibility="collapsed")
            if upload is not None:
                try:
                    image_data = upload.getvalue()
                    image = decode_upload(image_data)
                    st.image(image, caption=f"{upload.name} · {image.width} × {image.height} pixels")
                except ValueError as error:
                    st.error(str(error))
            else:
                st.markdown('<div class="empty-preview"><strong>Your handwriting goes here</strong><span>Choose a clear image of a single text line<br>to preview it before recognition.</span></div>', unsafe_allow_html=True)

            st.markdown("#### Optional preprocessing")
            st.caption("All off = baseline. Enabled techniques run in the order listed below.")
            enabled = [key for key, label in PREPROCESSING_LABELS.items()
                       if st.checkbox(label, value=False, key=f"preprocessing_{key}",
                                      disabled=image is None)]
            pipeline = build_preprocessing(enabled)
            preprocessing_config = pipeline.to_config()
            st.caption("Order: " + (" → ".join(PREPROCESSING_LABELS[key] for key in enabled)
                                    or "None (baseline)"))
            st.caption("These are demonstration defaults, not validated optimal settings. "
                       "The model still applies its required input preparation, including "
                       "CNN grayscale and height normalization.")
            with st.expander("Preprocessing parameters"):
                st.json(preprocessing_config)
            if image is not None:
                try:
                    with st.spinner("Preparing image…"):
                        processed_image = pipeline(image)
                    current_key = prediction_key(image_data, selected, token, preprocessing_config)
                    st.image(processed_image, caption=(
                        f"Model input before required preparation · {processed_image.width} × "
                        f"{processed_image.height} pixels · {processed_image.mode}"
                    ))
                except Exception as error:
                    st.error(f"Preprocessing failed: {error}")

    # A new image needs its own reference; changing only the model keeps it.
    upload_sha = hashlib.sha256(image_data).hexdigest() if image is not None else None
    if st.session_state.get("current_upload_sha") != upload_sha:
        st.session_state["correct_text"] = ""
        st.session_state["current_upload_sha"] = upload_sha

    # Changes to image, model, checkpoint or preprocessing invalidate predictions.
    previous = st.session_state.get("recognition_result")
    if previous is not None and previous["request_key"] != current_key:
        st.session_state.pop("recognition_result", None)

    with right:
        with st.container(border=True):
            st.markdown("### 2. Read and evaluate")
            st.caption(f"Selected model: {MODEL_LABELS[selected]}")
            ground_truth = st.text_area(
                "Correct text (optional)", height=110, key="correct_text",
                placeholder="Type exactly what is written in the image…",
                help="The model does not see this text. It is used only to calculate CER and WER after prediction.",
            )
            st.caption("Leave this blank for prediction only. You can add or edit it later without reading the image again.")
            submitted = st.button("Read handwriting", type="primary",
                                  disabled=processed_image is None or missing_checkpoint)
            if submitted and processed_image is not None:
                st.session_state.pop("recognition_result", None)
                try:
                    with st.spinner("Loading the selected model…"):
                        resource = get_model(selected, str(checkpoint), token)
                    with st.spinner("Reading your handwriting…"):
                        result = resource.recognize(processed_image)
                    st.session_state["recognition_result"] = {
                        **result, "request_key": current_key, "model_key": selected,
                        "image_sha256": hashlib.sha256(image_data).hexdigest(),
                        "file_name": upload.name,
                        "optional_preprocessing": preprocessing_config,
                    }
                except Exception as error:
                    st.error("The image could not be read. Check that the model is available and the image contains one readable text line.")
                    if selected == "trocr":
                        st.caption("TrOCR needs internet access to resolve/download its pinned checkpoint on first use.")
                    with st.expander("Error details"):
                        st.code(f"{type(error).__name__}: {error}", language=None)
            result = st.session_state.get("recognition_result")
            st.markdown("**Predicted text**")
            if result is None:
                st.info("Your model's prediction will appear here after you click Read handwriting.")
            elif result["prediction"]:
                st.code(result["prediction"], language=None)
            else:
                st.warning("The model returned an empty prediction for this image.")

    st.markdown("### 3. How close was the prediction?")
    result = st.session_state.get("recognition_result")
    scores = score_prediction(result["prediction"], ground_truth) if result is not None else None
    cer_card, wer_card, timing_card = st.columns(3)
    cer_card.metric("Character error rate", "Not calculated" if scores is None else f"{100 * scores['cer']:.2f}%")
    wer_card.metric("Word error rate", "Not calculated" if scores is None else f"{100 * scores['wer']:.2f}%")
    timing_card.metric("Reading time", "—" if result is None else f"{result['inference_seconds']:.2f} s")
    if result is not None and scores is None:
        st.caption("Add the correct text above to calculate CER and WER. An image alone cannot supply these scores.")
    else:
        st.caption("Lower error rates are better. Scoring normalizes whitespace and keeps case and punctuation. Rates can exceed 100%; they are not confidence scores.")

    if result is not None:
        download = {
            "file_name": result["file_name"], "image_sha256": result["image_sha256"],
            "model": result["model_config"], "prediction": result["prediction"],
            "ground_truth": ground_truth if ground_truth.strip() else None,
            "cer": None if scores is None else scores["cer"],
            "wer": None if scores is None else scores["wer"],
            "inference_seconds": result["inference_seconds"],
            "normalization": "collapse whitespace; retain case and punctuation",
            "input_adaptation": "EXIF orientation; alpha composited on white; RGB; optional preprocessing; mandatory model preparation",
            "optional_preprocessing": result["optional_preprocessing"],
            "purpose": "single uploaded image diagnostic",
        }
        st.download_button("Download this result", json.dumps(download, indent=2, ensure_ascii=False),
                           file_name="handwriting_result.json", mime="application/json")
        with st.expander("About this prediction"):
            config = result["model_config"]
            st.write("Model:", MODEL_LABELS[result["model_key"]])
            st.write("Optional preprocessing:", result["optional_preprocessing"])
            if config.get("checkpoint_epoch") is not None:
                st.write("Saved checkpoint epoch:", config["checkpoint_epoch"])
            st.caption("Reading time includes the recognizer's required image preparation, but excludes model loading and optional preprocessing. The supplied correct text is never passed to the recognizer.")

    st.markdown('<div class="footer-note">HTR Research Project · Uploaded images are processed in memory. Results are saved only when you download them.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
