"""Frontend helpers only: no Streamlit app, trained model, or downloads run."""
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from src.frontend.recognition import (
    MAX_UPLOAD_BYTES, TROCR_CHECKPOINT, TROCR_REVISION, RecognizerResource,
    PREPROCESSING_LABELS, build_preprocessing, checkpoint_token, decode_upload,
    load_resource, prediction_key, score_prediction,
)


def image_bytes(mode="RGB", color="white", size=(12, 6), format="PNG"):
    stream = BytesIO()
    Image.new(mode, size, color).save(stream, format=format)
    return stream.getvalue()


class FrontendRecognitionTests(unittest.TestCase):
    def test_preprocessing_baseline_preserves_decoded_upload(self):
        original = decode_upload(image_bytes(color=(80, 110, 150)))
        output = build_preprocessing()(original)
        self.assertIsNot(output, original)
        self.assertEqual(output.mode, original.mode)
        self.assertEqual(output.tobytes(), original.tobytes())
        self.assertEqual(build_preprocessing().to_config(), {"steps": []})

    def test_preprocessing_order_and_processed_input_reach_recognizer(self):
        original = decode_upload(image_bytes(color=(80, 110, 150)))
        before = original.tobytes()
        pipeline = build_preprocessing(reversed(list(PREPROCESSING_LABELS)))
        self.assertEqual([step["name"] for step in pipeline.to_config()["steps"]],
                         ["grayscale", "clahe", "non_local_means", "deskew",
                          "intensity_normalization", "size_normalization"])
        processed = pipeline(original)

        class Stub:
            def predict(self, image):
                self.image = image
                return "processed prediction"

            def to_config(self):
                return {}

        stub = Stub()
        self.assertEqual(RecognizerResource(stub).recognize(processed)["prediction"],
                         "processed prediction")
        self.assertEqual(stub.image.mode, "L")
        self.assertEqual(stub.image.size, (128, 64))
        self.assertEqual(stub.image.tobytes(), processed.tobytes())
        self.assertEqual(original.tobytes(), before)
        with self.assertRaises(ValueError):
            build_preprocessing(["unknown"])

    def test_preprocessing_changes_invalidate_prediction_identity(self):
        baseline = prediction_key(b"image", "trocr", "weights")
        for key in PREPROCESSING_LABELS:
            config = build_preprocessing([key]).to_config()
            self.assertNotEqual(baseline, prediction_key(b"image", "trocr", "weights", config))
        config = build_preprocessing(["size"]).to_config()
        first = prediction_key(b"image", "trocr", "weights", config)
        config["steps"][0]["target_height"] = 128
        self.assertNotEqual(first, prediction_key(b"image", "trocr", "weights", config))
        config = build_preprocessing(["grayscale", "clahe"]).to_config()
        first = prediction_key(b"image", "trocr", "weights", config)
        config["steps"].reverse()
        self.assertNotEqual(first, prediction_key(b"image", "trocr", "weights", config))

    def test_transparency_orientation_and_uploaded_bytes_preserved(self):
        data = image_bytes("RGBA", (0, 0, 0, 0))
        before = bytes(data)
        image = decode_upload(data)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(data, before)
        stream = BytesIO()
        original = Image.new("RGB", (12, 6), "white")
        exif = Image.Exif()
        exif[274] = 6
        original.save(stream, format="JPEG", exif=exif)
        self.assertEqual(decode_upload(stream.getvalue()).size, (6, 12))

    def test_invalid_oversized_unsupported_and_animated_uploads(self):
        for data in (b"", b"this is not an image", b"x" * (MAX_UPLOAD_BYTES + 1)):
            with self.assertRaises(ValueError):
                decode_upload(data)
        with self.assertRaisesRegex(ValueError, "PNG, JPG"):
            decode_upload(image_bytes(format="BMP"))
        with patch("src.frontend.recognition.MAX_IMAGE_PIXELS", 10):
            with self.assertRaisesRegex(ValueError, "megapixels"):
                decode_upload(image_bytes())
        stream = BytesIO()
        Image.new("RGB", (10, 10), "red").save(
            stream, format="PNG", save_all=True,
            append_images=[Image.new("RGB", (10, 10), "blue")], duration=100,
        )
        with self.assertRaisesRegex(ValueError, "still image"):
            decode_upload(stream.getvalue())

    def test_reference_required_and_rates_not_clamped(self):
        for reference in (None, "", "  \n "):
            self.assertIsNone(score_prediction("hello", reference))
        self.assertEqual(score_prediction("hello", "hello"), {"cer": 0, "wer": 0})
        self.assertEqual(score_prediction("", "hello"), {"cer": 1, "wer": 1})
        self.assertEqual(score_prediction("abc", "a")["cer"], 2)
        self.assertEqual(score_prediction("one two three", "one")["wer"], 2)
        self.assertEqual(score_prediction("Hello world!", " Hello\n  world! ")["cer"], 0)
        self.assertGreater(score_prediction("hello", "Hello")["cer"], 0)

    def test_cache_identity_tracks_image_model_and_checkpoint(self):
        key = prediction_key(b"image-a", "cnn_lstm_ctc", "weights-1")
        self.assertNotEqual(key, prediction_key(b"image-b", "cnn_lstm_ctc", "weights-1"))
        self.assertNotEqual(key, prediction_key(b"image-a", "trocr", "weights-1"))
        self.assertNotEqual(key, prediction_key(b"image-a", "cnn_lstm_ctc", "weights-2"))
        self.assertEqual(checkpoint_token("trocr"), TROCR_REVISION)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            self.assertTrue(checkpoint_token("cnn_lstm_ctc", path).startswith("missing:"))
            path.write_bytes(b"first")
            first = checkpoint_token("cnn_lstm_ctc", path)
            path.write_bytes(b"changed weights")
            self.assertNotEqual(first, checkpoint_token("cnn_lstm_ctc", path))
            with self.assertRaises(FileNotFoundError):
                load_resource("cnn_lstm_ctc", Path(directory) / "missing.pt")

    def test_resource_keeps_original_image_and_never_sees_reference(self):
        class Stub:
            def predict(self, image):
                image.putpixel((0, 0), (0, 0, 0))
                return "hello"

            def to_config(self):
                return {"architecture": "test stub"}

        original = Image.new("RGB", (12, 6), "white")
        result = RecognizerResource(Stub()).recognize(original)
        self.assertEqual(original.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(result["prediction"], "hello")
        self.assertGreaterEqual(result["inference_seconds"], 0)
        self.assertEqual(result["model_config"]["architecture"], "test stub")

    def test_loads_only_selected_factory_without_real_weights(self):
        import torch
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            path.write_bytes(b"placeholder, never actually loaded")
            with patch("src.models.CNNLSTMCTCRecognizer.from_checkpoint") as cnn, \
                    patch("src.models.TrOCRRecognizer") as trocr:
                resource = load_resource("cnn_lstm_ctc", path)
                self.assertIs(resource.recognizer, cnn.return_value)
                cnn.assert_called_once_with(path, device="cpu")
                trocr.assert_not_called()
            with patch("src.models.CNNLSTMCTCRecognizer.from_checkpoint") as cnn, \
                    patch("src.models.TrOCRRecognizer") as trocr:
                load_resource("trocr", path)
                cnn.assert_not_called()
                self.assertEqual(trocr.call_args.kwargs["checkpoint"], TROCR_CHECKPOINT)
                self.assertEqual(trocr.call_args.kwargs["revision"], TROCR_REVISION)
                self.assertFalse(trocr.call_args.kwargs["generation_kwargs"]["do_sample"])


if __name__ == "__main__":
    unittest.main()
