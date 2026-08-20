"""
common.py
Shared helpers used across Tasks 1-4.
Import these in the other task files, e.g.  from common import load_and_preprocess
"""

import numpy as np
from PIL import Image


def load_and_preprocess(path, size=(256, 256)):
    """Load an image from disk, convert to grayscale, resize to a common size.

    Returns a uint8 numpy array of shape (H, W).
    """
    img = Image.open(path).convert("L").resize(size)
    return np.array(img)
