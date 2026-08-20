# Assignment 3 - Hybrid Biomedical Image-Analysis Pipeline

Five per-task Python files. They share helpers in `common.py`.

## Setup
```bash
pip install ollama scikit-image matplotlib numpy pandas pillow torch torchvision
ollama pull llama3.2-vision
ollama pull llama3.2
# make sure the Ollama server/app is running before Tasks 1, 2, 4
```

Clone the dataset:
```bash
git clone https://github.com/Nickolay-K/Assingnment-3-dataset
```

## YOU MUST EDIT THESE PATHS
Every file has a path constant near the top pointing at `dataset/...`.
Point them at your actual clone. Check the repo's real folder layout first:
- `task1_vlm.py`  -> `IMAGE_GLOB`
- `task2_classical.py` -> `IMAGE_GLOB`
- `task3_unet.py` -> `IMG_DIR`, `MASK_DIR`
- `task4_pipeline.py` -> `TEST_GLOB`

CRITICAL for Task 3: after sorting, image `i` must correspond to mask `i`.
If filenames don't line up, the U-Net trains on mismatched pairs and Dice
stays near zero. Verify by eye before training.

## Run order
```bash
python task1_vlm.py        # EDA + VLM description (saves eda_*.png)
python task2_classical.py  # Otsu + regionprops + numbers-first LLM
python task3_unet.py       # trains U-Net, saves unet.pt + curves/panels
python task4_pipeline.py   # full pipeline -> pipeline_results.csv
```

## Notes / knobs to tune once you see YOUR data
- `min_size` / `area_threshold` in `task2_classical.classical_features` -
  morphological cleanup thresholds; depend on object scale in your modality.
- `density_class` thresholds in `task4_pipeline.py` - rule-of-thumb cutoffs.
- `epochs`, `lr`, batch size in `task3_unet.train_unet`.
- LLM outputs sometimes wrap JSON in ```json fences. Once you see what YOUR
  model returns, add a small `json.loads` with a fence-stripping fallback.

## If the Canvas U-Net skeleton is tested against
Replace the `UNet` class in `task3_unet.py` with theirs; keep the dataset,
metrics, losses, training loop, and visualisation.

## Extra credit hooks already present
- Loss ablation: `bce_loss`, `dice_loss`, `bce_dice` in `task3_unet.py` -
  pass each as `loss_fn=` to `train_unet` and compare validation Dice.
