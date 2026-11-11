"""Grayscale preserves existing luminance and original images."""
import unittest
import numpy as np
from PIL import Image
from src.preprocessing import Grayscale, PreprocessingPipeline


class GrayscaleTests(unittest.TestCase):
    def test_luminance_is_unchanged(self):
        image = Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8))
        result = PreprocessingPipeline((Grayscale(),))(image)
        np.testing.assert_array_equal(result, image)
        self.assertIsNot(result, image)

    def test_rgb_conversion_and_disabled(self):
        image = Image.new("RGB", (8, 8), (255, 0, 0))
        result = Grayscale()(image)
        self.assertEqual(result.mode, "L")
        self.assertEqual(result.getpixel((0, 0)), 76)
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(Grayscale(enabled=False)(image).mode, "RGB")
