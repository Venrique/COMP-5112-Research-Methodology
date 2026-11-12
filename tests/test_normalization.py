"""Normalization arithmetic, independence, configuration and source integrity."""
import json
import unittest

import numpy as np
from PIL import Image

from src.preprocessing import IntensityNormalization, SizeNormalization, PreprocessingPipeline
from src.validation.intensity_normalization import configurations as intensity_configs
from src.validation.size_normalization import configurations as size_configs


class NormalizationTests(unittest.TestCase):
    def test_intensity_exact_values_and_source_preserved(self):
        image = Image.fromarray(np.array([[50, 100, 150]], dtype=np.uint8))
        before = image.tobytes()
        result = IntensityNormalization()(image)
        self.assertEqual(np.asarray(result).tolist(), [[0, 128, 255]])
        self.assertEqual(result.size, image.size)
        self.assertEqual(image.tobytes(), before)

    def test_rgb_uses_single_global_range(self):
        image = Image.new('RGB', (2, 3), (50, 100, 150))
        result = IntensityNormalization()(image)
        self.assertEqual(result.mode, 'RGB')
        self.assertEqual(result.getpixel((0, 0)), (0, 128, 255))

    def test_constant_and_full_range(self):
        for values in ([42, 42, 42], [0, 128, 255]):
            image = Image.fromarray(np.array([values], dtype=np.uint8))
            result = IntensityNormalization()(image)
            self.assertEqual(result.tobytes(), image.tobytes())
            self.assertIsNot(result, image)

    def test_size_and_aspect_ratio_rounding(self):
        for mode in ('L', 'RGB'):
            image = Image.new(mode, (101, 20), 100)
            before = image.tobytes()
            result = SizeNormalization(target_height=32)(image)
            self.assertEqual(result.size, (162, 32))
            self.assertEqual(result.mode, mode)
            self.assertEqual(image.size, (101, 20))
            self.assertEqual(image.tobytes(), before)
        self.assertEqual(SizeNormalization(target_height=1)(Image.new('L', (1, 100))).size, (1, 1))

    def test_disabled_and_unsupported_modes(self):
        image = Image.new('RGBA', (5, 5))
        for cls in (IntensityNormalization, SizeNormalization):
            with self.assertRaises(ValueError):
                cls()(image)
            result = cls(enabled=False)(image)
            self.assertEqual(result.tobytes(), image.tobytes())
            self.assertIsNot(result, image)
            with self.assertRaises(ValueError):
                cls(enabled=1)
        for height in (0, -1, True, 2.5, '64'):
            with self.assertRaises(ValueError):
                SizeNormalization(target_height=height)

    def test_pipeline_independence_and_configs(self):
        image = Image.fromarray(np.array([[50, 100, 150]], dtype=np.uint8))
        intensity, size = IntensityNormalization(), SizeNormalization(target_height=2)
        self.assertEqual(PreprocessingPipeline((intensity,))(image).size, image.size)
        low, high = PreprocessingPipeline((size,))(image).getextrema()
        self.assertGreater(low, 0)
        self.assertLess(high, 255)
        combined = PreprocessingPipeline((intensity, size))
        self.assertEqual(combined(image).tobytes(), size(intensity(image)).tobytes())
        self.assertEqual(json.loads(json.dumps(combined.to_config()))['steps'][1]['target_height'], 2)
        self.assertEqual(set(intensity_configs()), {'baseline', 'intensity_normalization'})
        configs = size_configs()
        self.assertIsNone(configs.pop('baseline'))
        self.assertEqual({s.target_height for s in configs.values()}, {32, 48, 64, 96, 128})
        self.assertTrue(all(isinstance(s, SizeNormalization) for s in configs.values()))
