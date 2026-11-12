"""Controls for the grayscale validation comparison."""
import unittest
from PIL import Image
from src.validation.grayscale import configurations
from src.validation.denoising import adapt_image


class GrayscaleValidationTests(unittest.TestCase):
    def test_existing_luminance_is_identical_at_model_input(self):
        image = Image.new('L', (4, 4), 123)
        gray = configurations()['grayscale'](adapt_image(image))
        self.assertEqual(gray.convert('RGB').tobytes(), adapt_image(image).convert('RGB').tobytes())

    def test_alpha_handling_is_shared_and_sources_are_preserved(self):
        source = Image.new('RGBA', (4, 4), (200, 50, 10, 128))
        pixels = source.tobytes()
        baseline = adapt_image(source)
        gray = configurations()['grayscale'](baseline)
        self.assertEqual(gray.mode, 'L')
        self.assertEqual(gray.tobytes(), baseline.convert('L').tobytes())
        self.assertEqual(source.tobytes(), pixels)
        self.assertEqual(set(configurations()), {'baseline', 'grayscale'})
