"""
task2_classical.py
Task 2 - Classical feature extraction (Otsu + regionprops) and a numbers-only
LLM interpretation. The LLM in this task NEVER sees the image, only the numbers.

Run:
    python task2_classical.py
"""

import glob
import numpy as np
import pandas as pd
from skimage import filters, morphology, measure
import ollama

from common import load_and_preprocess

IMAGE_GLOB = "dataset/images/*"   # <-- EDIT to your path


def classical_features(gray_img):
    """Otsu threshold -> morphological cleanup -> label -> regionprops table.

    Returns (labels, props_dict). Wrap props_dict in pd.DataFrame for a table.
    """
    thresh = filters.threshold_otsu(gray_img)
    binary = gray_img > thresh
    binary = morphology.remove_small_objects(binary, min_size=64)
    binary = morphology.remove_small_holes(binary, area_threshold=64)
    labels = measure.label(binary)

    props = measure.regionprops_table(
        labels,
        intensity_image=gray_img,
        properties=("area", "eccentricity", "solidity", "mean_intensity"),
    )
    return labels, props


def summarise_features(props):
    """Turn the numeric table into a numbers-only natural-language summary."""
    n = len(props["area"])
    if n == 0:
        return "No objects were detected after thresholding and cleanup."
    return (
        f"{n} objects detected. "
        f"Mean area {np.mean(props['area']):.1f} px "
        f"(min {np.min(props['area']):.0f}, max {np.max(props['area']):.0f}). "
        f"Mean eccentricity {np.mean(props['eccentricity']):.2f}, "
        f"mean solidity {np.mean(props['solidity']):.2f}, "
        f"mean intensity {np.mean(props['mean_intensity']):.1f}."
    )


NUMBERS_PROMPT = """You are given ONLY numeric measurements from an image (you cannot see the image).
Write ONE short paragraph describing the objects, then return ONLY a JSON object with keys:
{"n_objects": <int>, "density_class": "low|medium|high",
 "shape_regularity": "regular|irregular|uncertain", "quality_flag": "ok|poor|uncertain"}

Measurements:
"""


def interpret_numbers(summary_text):
    """Pass the numbers-only summary to a local text LLM (no image)."""
    resp = ollama.chat(
        model="llama3.2",  # text model; never sees the image
        messages=[{"role": "user", "content": NUMBERS_PROMPT + summary_text}],
    )
    return resp["message"]["content"]


if __name__ == "__main__":
    paths = sorted(glob.glob(IMAGE_GLOB))
    if not paths:
        raise SystemExit(f"No images found at {IMAGE_GLOB} - edit IMAGE_GLOB.")

    gray = load_and_preprocess(paths[0])
    _, props = classical_features(gray)

    print("=== FEATURE TABLE ===")
    print(pd.DataFrame(props), "\n")

    summary = summarise_features(props)
    print("=== NUMBERS-ONLY SUMMARY ===")
    print(summary, "\n")

    print("=== LLM INTERPRETATION (numbers-first) ===")
    print(interpret_numbers(summary))
