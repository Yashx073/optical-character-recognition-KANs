# Optical Character Recognition with CNN, RNN, and KAN

This project is a handwritten **A–Z character recognition** system built with TensorFlow/Keras and exposed through a Streamlit app.

It includes:
- a CNN classifier
- an RNN classifier
- a hybrid CNN feature extractor + KAN classifier
- notebooks for training/experimentation
- pretrained model files for quick inference

---

## What this repository contains

- **Inference app**: `app.py`
- **Pretrained models**: `models/`
  - `cls_deneme_model.h5` (CNN)
  - `rnn_ocr_model.h5` / `rnn_ocr_model.keras` (RNN)
  - `feature_extractor.h5` + `kan_model.h5` (CNN+KAN pipeline)
- **Training / experimentation notebooks**:
  - `OCR.ipynb`
  - `test.ipynb`
  - `kan.ipynb`
- **Sample inputs**: `test_images/`
- **Additional visual assets**: `figures/`, `Major Project photos/`

---

## Features

- Upload handwritten character images (`.png`, `.jpg`, `.jpeg`)
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
prime-run streamlit run app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`).

---

## How inference works

1. App loads available models from `models/`
2. Uploaded image is preprocessed into model-ready input
3. Prediction is run through each loaded model
4. Numeric class index is mapped to uppercase letters (`0 -> A`, `25 -> Z`)
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
