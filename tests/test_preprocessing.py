"""Synthetic checks; these do not measure handwriting recognition accuracy."""

import json
import unittest

import numpy as np
from PIL import Image

from src.preprocessing import NonLocalMeans, PreprocessingPipeline


class PreprocessingTests(unittest.TestCase):
    def test_baseline_and_disabled_preserve_pixels_and_mode(self):
        for mode, shape in (("L", (32, 64)), ("RGB", (32, 64, 3)), ("RGBA", (32, 64, 4))):
            image = Image.fromarray(np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8))
            for pipeline in (PreprocessingPipeline(), PreprocessingPipeline((NonLocalMeans(enabled=False),))):
                result = pipeline(image)
                self.assertIsNot(result, image)
                self.assertEqual(result.mode, mode)
                np.testing.assert_array_equal(result, image)
                result.putpixel((0, 0), 0)
                self.assertNotEqual(result.getpixel((0, 0)), image.getpixel((0, 0)))

    def test_denoising_reduces_synthetic_noise_and_preserves_input(self):
        for shape in ((48, 96), (48, 96, 3)):
            pixels = np.clip(128 + np.random.default_rng(2).normal(0, 6, shape), 0, 255).astype(np.uint8)
            image = Image.fromarray(pixels)
            pipeline = PreprocessingPipeline((NonLocalMeans(h=12, h_color=12),))
            result = pipeline(image)
            self.assertEqual(result.mode, image.mode)
            self.assertEqual(result.size, image.size)
            self.assertEqual(np.asarray(result).dtype, np.uint8)
            self.assertLess(np.mean((np.asarray(result).astype(float) - 128) ** 2), np.mean((pixels.astype(float) - 128) ** 2))
            np.testing.assert_array_equal(image, pixels)
            np.testing.assert_array_equal(result, pipeline(image))

    def test_rgb_channels_are_not_swapped(self):
        image = Image.new("RGB", (32, 32), (220, 50, 10))
        result = np.asarray(NonLocalMeans()(image)).mean(axis=(0, 1))
        np.testing.assert_allclose(result, (220, 50, 10), atol=5)

    def test_invalid_parameters_and_modes(self):
        for kwargs in ({"h": -1}, {"h": float("nan")}, {"h_color": 0}, {"template_window_size": 2}, {"search_window_size": 3}, {"enabled": "yes"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NonLocalMeans(**kwargs)
        with self.assertRaises(ValueError):
            NonLocalMeans()(Image.new("RGBA", (32, 32)))

    def test_configuration_is_serializable_and_order_is_preserved(self):
        steps = (NonLocalMeans(enabled=False), NonLocalMeans(h=8))
        config = json.loads(json.dumps(PreprocessingPipeline(steps).to_config()))
        self.assertFalse(config["steps"][0]["enabled"])
        self.assertEqual(config["steps"][1]["h"], 8)

    def test_pipeline_executes_enabled_steps_in_order(self):
        calls = []

        class Step:
            def __init__(self, name, enabled=True):
                self.name, self.enabled = name, enabled

            def __call__(self, image):
                calls.append(self.name)
                return image

        PreprocessingPipeline((Step("first"), Step("skip", False), Step("last")))(Image.new("L", (2, 2)))
        self.assertEqual(calls, ["first", "last"])


if __name__ == "__main__":
    unittest.main()
