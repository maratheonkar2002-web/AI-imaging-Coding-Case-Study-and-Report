"""
task4_pipeline.py
Task 4 - Hybrid pipeline on the unseen test images:
    raw image -> U-Net mask -> regionprops feature table
    -> structured JSON record -> one-paragraph narrative
    -> aggregated pandas DataFrame saved as CSV.

Run (after training in task3):
    python task4_pipeline.py
Requires unet.pt (produced by task3_unet.py) and your test images.
"""

import os
import glob
import numpy as np
import pandas as pd
from skimage import measure
import torch

from common import load_and_preprocess
from task2_classical import summarise_features, interpret_numbers
from task3_unet import UNet

TEST_GLOB = "dataset/test/*"   # <-- EDIT
MODEL_PATH = "unet.pt"


def density_class(n, area_frac):
    """Rule mapping object count / coverage fraction to a label."""
    if n == 0:
        return "none"
    if area_frac > 0.15 or n > 20:
        return "high"
    if area_frac > 0.05 or n > 5:
        return "medium"
    return "low"


def pipeline_one_image(model, img_path, image_id, size=(256, 256), device=None):
    """Run the full pipeline on one image; return (json_record, narrative)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gray = load_and_preprocess(img_path, size)
    x = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(model(x.to(device)))[0, 0].cpu().numpy() > 0.5)

    labels = measure.label(pred)
    props = measure.regionprops_table(
        labels, intensity_image=gray,
        properties=("area", "eccentricity", "solidity", "mean_intensity"))
    n = len(props["area"])
    mean_area = float(np.mean(props["area"])) if n else 0.0
    area_frac = float(pred.sum()) / (size[0] * size[1])

    # The JSON record is the auditable "source of truth" for this image.
    record = {
        "image_id": image_id,
        "n_objects": int(n),
        "mean_area": round(mean_area, 1),
        "density_class": density_class(n, area_frac),
        "quality_flag": "ok" if n > 0 else "poor",
    }

    # Narrative comes from the numbers-only LLM (it never sees the image).
    summary = summarise_features(props)
    narrative = interpret_numbers(summary)
    return record, narrative


def run_pipeline(model, test_paths, device=None):
    """Run the pipeline across all test images and aggregate to a CSV."""
    records = []
    for p in test_paths:
        rec, narr = pipeline_one_image(model, p, os.path.basename(p), device=device)
        print(rec)
        print(narr, "\n")
        records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv("pipeline_results.csv", index=False)
    print("saved pipeline_results.csv")
    return df


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    test_paths = sorted(glob.glob(TEST_GLOB))
    if not test_paths:
        raise SystemExit(f"No test images at {TEST_GLOB} - edit TEST_GLOB.")

    df = run_pipeline(model, test_paths, device=device)
    print(df)
