"""Five CLAHE contrasts using the same runner as prior validation studies.

Labels expose clip limit and tile grid dimensions (numbers of tiles, not pixels).
"""
from src.preprocessing.contrast import CLAHE
from src.validation.denoising import evaluate, summarize


def configurations() -> dict:
    """Vary one parameter relative to clip_limit=2, tile_grid_size=(8, 8)."""
    return {
        'baseline': None,
        'clahe_clip1_grid8x8': CLAHE(clip_limit=1, tile_grid_size=(8, 8)),
        'clahe_clip2_grid8x8': CLAHE(clip_limit=2, tile_grid_size=(8, 8)),
        'clahe_clip4_grid8x8': CLAHE(clip_limit=4, tile_grid_size=(8, 8)),
        'clahe_clip2_grid4x4': CLAHE(clip_limit=2, tile_grid_size=(4, 4)),
        'clahe_clip2_grid16x16': CLAHE(clip_limit=2, tile_grid_size=(16, 16)),
    }
