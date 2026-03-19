# Optical Character Recognition with CNN, RNN, and KAN

This project is a handwritten **multi-script character recognition** system built with TensorFlow/Keras and exposed through a Streamlit app.

It includes:
- a CNN classifier
- an RNN classifier
- a hybrid CNN feature extractor + KAN classifier
- support for English (A–Z) and Devanagari script selection in the app
- notebooks for training/experimentation
- pretrained model files for quick inference

---

## What this repository contains

- **Inference app**: `app.py`
- **Devanagari training script**: `train_devanagari.py`
- **Pretrained models**: `models/`
  - `cls_deneme_model.h5` (CNN)
  - `rnn_ocr_model.h5` / `rnn_ocr_model.keras` (RNN)
  - `feature_extractor.h5` + `kan_model.h5` (CNN+KAN pipeline)
  - optional Devanagari model files:
    - `devanagari_cnn_model.h5`
    - `devanagari_rnn_model.h5`
    - `devanagari_feature_extractor.h5`
    - `devanagari_kan_model.h5`
    - `devanagari_labels.txt`
  - KAN notebook artifacts:
    - `cnn_model.keras`, `feature_extractor.keras`, `kan_model.keras`
    - `kan_calibration.npz`
- **Training / experimentation notebooks**:
  - `OCR.ipynb`
  - `test.ipynb`
  - `kan.ipynb`
- **Sample inputs**: `test_images/`
- **Additional visual assets**: `figures/`, `Major Project photos/`

---

## Features

- Upload handwritten character images (`.png`, `.jpg`, `.jpeg`)
- Switch inference script between **English (A–Z)** and **Devanagari**
- Automatic preprocessing to MNIST-like format:
  - grayscale
  - resize to `28x28`
  - Gaussian blur
  - adaptive thresholding
  - normalization (0–1) and optional inversion
- Side-by-side predictions from available models:
  - CNN
  - RNN
  - CNN+KAN
- Confidence score display for each model

---

## Requirements

### Core runtime dependencies (for `app.py`)

- Python 3.9+
- streamlit
- tensorflow
- numpy
- opencv-python

Install:

```bash
pip install streamlit tensorflow numpy opencv-python
```

### Notebook/training dependencies (optional)

From notebook cells, additional packages used include:
- pandas
- seaborn
- matplotlib
- scikit-learn
- torch
- pyyaml
- tqdm
- pykan

Optional install:

```bash
pip install pandas seaborn matplotlib scikit-learn torch pyyaml tqdm
pip install git+https://github.com/KindXiaoming/pykan.git
```

---

## Run the app

From the project root:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`).

---

## Devanagari integration (your dataset)

Dataset layout expected by the training script:

```text
DEVNAGARI_NEW/
  TRAIN/
    1/
    2/
    ...
  TEST/
    1/
    2/
    ...
```

Train all Devanagari models (CNN, RNN, and CNN+KAN):

```bash
python train_devanagari.py --data-dir DEVNAGARI_NEW --cnn-epochs 10 --rnn-epochs 10 --kan-epochs 12
```

This saves:
- `models/devanagari_cnn_model.h5`
- `models/devanagari_rnn_model.h5`
- `models/devanagari_feature_extractor.h5`
- `models/devanagari_kan_model.h5`
- `models/devanagari_labels.txt`

In the Streamlit app, select **Devanagari** from the script dropdown. The app will automatically use these files when available.

---

## Current status

- ✅ `app.py` supports script switching (**English (A–Z)** / **Devanagari**) and loads both `.h5` and `.keras` model files when available.
- ✅ `train_devanagari.py` trains CNN, RNN, and CNN+KAN models and exports Devanagari artifacts used by the app.
- ✅ `kan.ipynb` contains improved CNN+KAN training with calibration/rejection logic and exports `.keras` models + calibration settings.
- ℹ️ On this Linux setup, TensorFlow currently falls back to CPU with the warning `Cannot dlopen some GPU libraries`; training still runs successfully.

---

## How inference works

1. App loads available models from `models/` for the selected script
2. Uploaded image is preprocessed into model-ready input
3. Prediction is run through each loaded model
4. Numeric class index is mapped to script labels
   - English: `0 -> A`, `25 -> Z`
   - Devanagari: loaded from `models/devanagari_labels.txt` when present, otherwise folder-based labels (`Class 1`, `Class 2`, ...)
5. Top prediction + confidence is shown

If a model file is missing or incompatible, the app skips that model and shows a warning.

---

## Dataset references

The notebooks reference the A–Z handwritten dataset:
- https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format/data

Related notebook reference code:
- https://www.kaggle.com/code/mahmutyldrmm/a-z-handwritting-alphabets-project

---

## Notes

- `KANLayer` is implemented in `app.py` and loaded via `custom_object_scope` for Keras model deserialization.
- Some notebook cells contain historical outputs/errors; this does not affect running `app.py` with valid model files.
# optical-character-recognition-KANs
