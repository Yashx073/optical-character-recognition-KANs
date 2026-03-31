import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.utils import custom_object_scope


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;600;700&display=swap');

        .stApp {
            background:
                radial-gradient(1200px 500px at 15% -10%, rgba(14, 165, 233, 0.20), transparent 60%),
                radial-gradient(1000px 450px at 85% 0%, rgba(34, 197, 94, 0.18), transparent 60%),
                linear-gradient(180deg, #020617 0%, #0B1220 55%, #0A101C 100%);
            color: #E5E7EB;
        }

        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: 0.2px;
        }

        body, p, li, .stMarkdown, .stCaption, .stAlert {
            font-family: 'Manrope', sans-serif;
        }

        .hero {
            padding: 1.1rem 1.2rem;
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: linear-gradient(135deg, rgba(2, 132, 199, 0.18), rgba(34, 197, 94, 0.14));
            box-shadow: 0 12px 24px rgba(2, 6, 23, 0.28);
            margin-bottom: 0.8rem;
        }

        .hero-title {
            margin: 0;
            font-size: 2rem;
            line-height: 1.15;
            color: #F8FAFC;
            font-weight: 700;
        }

        .hero-subtitle {
            margin: 0.55rem 0 0 0;
            color: #CBD5E1;
            font-size: 0.99rem;
        }

        .section-label {
            margin: 1.1rem 0 0.5rem 0;
            color: #93C5FD;
            font-size: 0.86rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }

        .meta-chip {
            display: inline-block;
            margin-top: 0.35rem;
            padding: 0.36rem 0.65rem;
            border-radius: 999px;
            background: rgba(30, 41, 59, 0.65);
            border: 1px solid rgba(148, 163, 184, 0.22);
            color: #BFDBFE;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .result-card {
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: linear-gradient(160deg, rgba(15, 23, 42, 0.96), rgba(17, 24, 39, 0.9));
            padding: 0.8rem 0.85rem;
            margin-bottom: 0.55rem;
            box-shadow: 0 6px 16px rgba(2, 6, 23, 0.25);
        }

        .result-model {
            color: #7DD3FC;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 0;
            font-weight: 700;
        }

        .result-pred {
            color: #F8FAFC;
            margin: 0.22rem 0;
            font-size: 1.05rem;
            font-weight: 700;
        }

        .result-conf {
            margin: 0;
            color: #86EFAC;
            font-size: 0.86rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <div class="hero">
            <h1 class="hero-title">Optical Character Recognition Studio</h1>
            <p class="hero-subtitle">Upload a handwritten character to compare CNN, RNN, and CNN+KAN predictions in one place.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_cards(results):
    for model_name, label, confidence in results:
        st.markdown(
            f"""
            <div class="result-card">
                <p class="result-model">{model_name}</p>
                <p class="result-pred">Prediction: {label}</p>
                <p class="result-conf">Confidence: {confidence * 100:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==========================================
# 1️⃣ Define KANLayer (Fixed for Serialization)
# ==========================================
class KANLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )
        self.alpha = self.add_weight(
            shape=(self.units,),
            initializer="ones",
            trainable=True
        )

    def call(self, inputs):
        linear = tf.matmul(inputs, self.W) + self.b
        nonlinear = tf.math.sin(self.alpha * linear)
        return linear + nonlinear

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


SCRIPT_CONFIGS = {
    "English (A–Z)": {
        "model_prefix": "",
        "title": "English (A–Z)",
    },
    "Devanagari": {
        "model_prefix": "devanagari_",
        "title": "Devanagari",
    },
}


def list_class_dirs(path):
    if not os.path.isdir(path):
        return []

    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    numeric_folders = [folder for folder in folders if folder.isdigit()]
    return sorted(numeric_folders, key=lambda value: int(value))


def load_labels(script_name):
    if script_name == "English (A–Z)":
        return [chr(index + 65) for index in range(26)]

    label_files = [
        os.path.join("models", "devanagari_labels.txt"),
        os.path.join("DEVNAGARI_NEW", "labels.txt"),
    ]
    for label_file in label_files:
        if os.path.exists(label_file):
            with open(label_file, "r", encoding="utf-8") as label_handle:
                labels = [line.strip() for line in label_handle if line.strip()]
            if labels:
                return labels

    train_class_dirs = list_class_dirs(os.path.join("DEVNAGARI_NEW", "TRAIN"))
    if train_class_dirs:
        return [f"Class {folder}" for folder in train_class_dirs]

    return []


def model_path_candidates(model_prefix, base_name):
    return [
        os.path.join("models", f"{model_prefix}{base_name}.h5"),
        os.path.join("models", f"{model_prefix}{base_name}.keras"),
    ]


# ==========================================
# 2️⃣ Streamlit UI Setup
# ==========================================
st.set_page_config(page_title="OCR Character Recognition", layout="centered")
inject_styles()
render_header()

st.markdown('<p class="section-label">Script Selection</p>', unsafe_allow_html=True)
script_name = st.selectbox("📝 Select script", list(SCRIPT_CONFIGS.keys()))
class_labels = load_labels(script_name)

if script_name == "Devanagari":
    if not class_labels:
        st.warning("⚠️ No Devanagari class labels found. Add `models/devanagari_labels.txt` or keep dataset folders in `DEVNAGARI_NEW/TRAIN/`.")

st.caption(f"Active script: {SCRIPT_CONFIGS[script_name]['title']}")
st.markdown(
    f'<span class="meta-chip">Active Script: {SCRIPT_CONFIGS[script_name]["title"]}</span>',
    unsafe_allow_html=True,
)

# ==========================================
# 3️⃣ Load Models Safely with Custom Scope
# ==========================================
@st.cache_resource
def load_all_models(script_name):
    models_dict = {}
    model_prefix = SCRIPT_CONFIGS[script_name]["model_prefix"]

    def safe_load_model(name, paths, custom_objects=None):
        model_path = next((path for path in paths if os.path.exists(path)), None)
        if model_path is None:
            st.warning(f"⚠️ Could not find {name} model file. Checked: {', '.join(paths)}")
            return None

        try:
            if custom_objects:
                with custom_object_scope(custom_objects):
                    model = tf.keras.models.load_model(model_path, compile=False)
            else:
                model = tf.keras.models.load_model(model_path, compile=False)
            return model
        except Exception as e:
            st.warning(f"⚠️ Could not load {name} model ({model_path}): {e}")
            return None

    if script_name == "English (A–Z)":
        cnn_candidates = [os.path.join("models", "cls_deneme_model.h5"), os.path.join("models", "cls_deneme_model.keras"), os.path.join("models", "cnn_model.h5"), os.path.join("models", "cnn_model.keras")]
        rnn_candidates = model_path_candidates(model_prefix, "rnn_ocr_model")
    else:
        cnn_candidates = model_path_candidates(model_prefix, "cnn_model")
        rnn_candidates = model_path_candidates(model_prefix, "rnn_model")

    feat_candidates = model_path_candidates(model_prefix, "feature_extractor")
    kan_candidates = model_path_candidates(model_prefix, "kan_model")

    # Load models
    models_dict["CNN"] = safe_load_model("CNN", cnn_candidates)
    models_dict["RNN"] = safe_load_model("RNN", rnn_candidates)
    feature_extractor = safe_load_model("Feature Extractor", feat_candidates)
    kan_model = safe_load_model("KAN Classifier", kan_candidates, {"KANLayer": KANLayer})

    if feature_extractor and kan_model:
        models_dict["CNN+KAN"] = (feature_extractor, kan_model)

    return models_dict


models_dict = load_all_models(script_name)
st.write("---")

# ==========================================
# 4️⃣ Upload and Preprocess Image (MNIST Style)
# ==========================================
st.markdown('<p class="section-label">Input Image</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload a handwritten character image", type=["png", "jpg", "jpeg"])


def decode_prediction(pred, labels):
    class_index = int(np.argmax(pred, axis=1)[0])
    confidence = float(np.max(pred))
    if 0 <= class_index < len(labels):
        label = labels[class_index]
    else:
        label = f"Class {class_index}"
    return label, confidence

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        st.error("❌ Could not read image — please try another file.")
        st.stop()

    # 🧹 Step 1: Resize to MNIST shape
    img = cv2.resize(img, (28, 28))

    # 🧹 Step 2: Gaussian blur to reduce background noise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # 🧹 Step 3: Adaptive threshold to remove grey lines
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # 🧹 Step 4: Normalize to 0–1
    img_norm = img.astype("float32") / 255.0

    # 🧹 Step 5 (optional): Auto-invert if background is light
    if np.mean(img_norm) > 0.5:
        img_norm = 1 - img_norm

    preview_col, note_col = st.columns([1.1, 1.3])
    with preview_col:
        st.image(img, caption="Preprocessed (MNIST-style) Image", width=170)
    with note_col:
        st.markdown('<p class="section-label">Preprocessing</p>', unsafe_allow_html=True)
        st.caption("Resized to 28x28, denoised, thresholded, and normalized for model inference.")

    st.write("---")

    # ==========================================
    # 5️⃣ Prediction Functions
    # ==========================================
    def predict_cnn(model, img):
        x = img.reshape(1, 28, 28, 1)
        pred = model.predict(x, verbose=0)
        return decode_prediction(pred, class_labels)

    def predict_rnn(model, img):
        x = img.reshape(1, 28, 28)
        pred = model.predict(x, verbose=0)
        return decode_prediction(pred, class_labels)

    def predict_cnn_kan(feature_extractor, kan_model, img):
        x = img.reshape(1, 28, 28, 1)
        features = feature_extractor.predict(x, verbose=0)
        pred = kan_model.predict(features, verbose=0)
        return decode_prediction(pred, class_labels)

    # ==========================================
    # 6️⃣ Run Predictions
    # ==========================================
    st.markdown('<p class="section-label">Model Predictions</p>', unsafe_allow_html=True)
    results = []

    try:
        if models_dict.get("CNN"):
            label, conf = predict_cnn(models_dict["CNN"], img_norm)
            results.append(("CNN", label, conf))
    except Exception as e:
        st.error(f"❌ CNN prediction failed: {e}")

    try:
        if models_dict.get("RNN"):
            label, conf = predict_rnn(models_dict["RNN"], img_norm)
            results.append(("RNN", label, conf))
    except Exception as e:
        st.error(f"❌ RNN prediction failed: {e}")

    try:
        if models_dict.get("CNN+KAN"):
            feat, kan = models_dict["CNN+KAN"]
            label, conf = predict_cnn_kan(feat, kan, img_norm)
            results.append(("CNN+KAN", label, conf))
    except Exception as e:
        st.error(f"❌ CNN+KAN prediction failed: {e}")

    # ==========================================
    # 7️⃣ Show Results
    # ==========================================
    if results:
        render_result_cards(results)
    else:
        st.warning("⚠️ No predictions could be made. Check if models are loaded correctly.")
else:
    st.info("👆 Upload an image above to start recognition.")

# ==========================================
# 8️⃣ Footer
# ==========================================
st.markdown("---")