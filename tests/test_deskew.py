"""Geometry checks, not evidence of recognition improvement."""
import json
import unittest
import numpy as np
from PIL import Image, ImageDraw
from src.preprocessing import Deskew, PreprocessingPipeline


class DeskewTests(unittest.TestCase):
    def test_corrects_both_directions_without_mutation(self):
        original = Image.new("L", (300, 80), 255)
        draw = ImageDraw.Draw(original)
        for x in range(20, 270, 20):
            draw.rectangle((x, 30, x + 12, 44), fill=0)
        for mode in ("L", "RGB"):
            for angle in (-7, 7):
                image = original.convert(mode).rotate(angle, expand=True, fillcolor=255 if mode == "L" else (255, 255, 255))
                before = np.array(image)
                step = Deskew()
                self.assertAlmostEqual(step.estimate_angle(image), -angle, delta=0.5)
                result = PreprocessingPipeline((step,))(image)
                self.assertEqual(result.mode, mode)
                self.assertGreaterEqual(result.width, image.width)
                self.assertGreaterEqual(result.height, image.height)
                self.assertAlmostEqual(step.estimate_angle(result), 0, delta=0.5)
                np.testing.assert_array_equal(image, before)
                np.testing.assert_array_equal(result, step(image))

    def test_blank_disabled_and_config(self):
        for mode in ("L", "RGB"):
            image = Image.new(mode, (60, 20), "white")
            self.assertEqual(Deskew().estimate_angle(image), 0)
            np.testing.assert_array_equal(Deskew()(image), image)
        image = Image.new("RGBA", (60, 20))
        result = Deskew(enabled=False)(image)
        self.assertIsNot(result, image)
        np.testing.assert_array_equal(result, image)
        config = json.loads(json.dumps(PreprocessingPipeline((Deskew(),)).to_config()))
        self.assertEqual(config["steps"][0]["name"], "deskew")
        with self.assertRaises(ValueError):
            Deskew()(image)

    def test_invalid_parameters(self):
        for kwargs in ({"enabled": 1}, {"max_angle": 0}, {"max_angle": 46},
                       {"angle_step": float("nan")}, {"angle_step": 11},
                       {"max_angle": True}, {"angle_step": 0.0001}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Deskew(**kwargs)
