"""
task1_vlm.py
Task 1 - Data preparation, EDA, and multimodal LLM (llama3.2-vision) description.

Setup:
    pip install ollama scikit-image matplotlib numpy pandas pillow
    ollama pull llama3.2-vision
    (make sure the Ollama server/app is running)

Run:
    python task1_vlm.py
Adjust IMAGE_GLOB below to point at your cloned dataset.
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import ollama

from common import load_and_preprocess

IMAGE_GLOB = "dataset/images/*"   # <-- EDIT to your path


# ---------- Task 1a: EDA ----------
def build_eda(image_paths, size=(256, 256), n_show=4):
    """Show a sample of preprocessed images and a pooled intensity histogram.

    Saves eda_samples.png and eda_histogram.png for the report.
    """
    imgs = [load_and_preprocess(p, size) for p in image_paths]

    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
    for ax, im in zip(axes, imgs[:n_show]):
        ax.imshow(im, cmap="gray")
        ax.axis("off")
    plt.suptitle("Sample preprocessed images")
    plt.tight_layout()
    plt.savefig("eda_samples.png", dpi=150)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(np.concatenate([im.ravel() for im in imgs]), bins=50)
    plt.xlabel("Pixel intensity")
    plt.ylabel("Count")
    plt.title("Pooled intensity histogram")
    plt.savefig("eda_histogram.png", dpi=150)
    plt.show()
    return imgs


# ---------- Task 1b: VLM description ----------
NAIVE_PROMPT = "What is in this medical image?"

# Engineered prompt: anchors descriptive-not-diagnostic, forces JSON, permits "uncertain".
STRUCTURED_PROMPT = """You are a descriptive image-analysis assistant, NOT a diagnostician.
Describe only what is visually present. Do NOT infer disease or give a diagnosis.
If any field is unclear, you MUST use the string "uncertain".

Return ONLY a valid JSON object, no prose, with exactly these keys:
{
  "modality": "...",
  "tissue_type": "...",
  "notable_features": "...",
  "image_quality": "..."
}"""


def describe_image(image_path, prompt):
    """Send one image to llama3.2-vision via Ollama and return the raw text."""
    resp = ollama.chat(
        model="llama3.2-vision",
        messages=[{"role": "user", "content": prompt, "images": [image_path]}],
    )
    return resp["message"]["content"]


def show_run_variability(image_path, prompt, n=3):
    """Same prompt, n runs -> demonstrates the outputs are not identical."""
    return [describe_image(image_path, prompt) for _ in range(n)]


if __name__ == "__main__":
    paths = sorted(glob.glob(IMAGE_GLOB))
    if not paths:
        raise SystemExit(f"No images found at {IMAGE_GLOB} - edit IMAGE_GLOB.")

    build_eda(paths)

    print("=== NAIVE PROMPT ===")
    print(describe_image(paths[0], NAIVE_PROMPT), "\n")

    print("=== STRUCTURED PROMPT ===")
    print(describe_image(paths[0], STRUCTURED_PROMPT), "\n")

    print("=== RUN-TO-RUN VARIABILITY (structured prompt x3) ===")
    for i, out in enumerate(show_run_variability(paths[0], STRUCTURED_PROMPT), 1):
        print(f"--- run {i} ---\n{out}\n")
