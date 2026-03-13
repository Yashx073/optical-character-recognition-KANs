import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.utils import custom_object_scope

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


# ==========================================
# 2️⃣ Streamlit UI Setup
# ==========================================
st.set_page_config(page_title="OCR Character Recognition (A–Z)", layout="centered")
st.title("🔠 Optical Character Recognition (A–Z)")
st.write("Upload a handwritten character image to test CNN, RNN, and CNN+KAN models.")

# ==========================================
# 3️⃣ Load Models Safely with Custom Scope
# ==========================================
@st.cache_resource
def load_all_models():
    models_dict = {}

    def safe_load_model(name, path, custom_objects=None):
        try:
            if custom_objects:
                with custom_object_scope(custom_objects):
                    model = tf.keras.models.load_model(path, compile=False)
            else:
                model = tf.keras.models.load_model(path, compile=False)
            st.success(f"✅ {name} model loaded: {os.path.basename(path)}")
            return model
        except Exception as e:
            st.warning(f"⚠️ Could not load {name} model ({path}): {e}")
            return None

    # Paths
    cnn_path = os.path.join("models", "cls_deneme_model.h5")
    rnn_path = os.path.join("models", "rnn_ocr_model.h5")
    feat_path = os.path.join("models", "feature_extractor.h5")
    kan_path = os.path.join("models", "kan_model.h5")

    # Load models
    models_dict["CNN"] = safe_load_model("CNN", cnn_path)
    models_dict["RNN"] = safe_load_model("RNN", rnn_path)
    feature_extractor = safe_load_model("Feature Extractor", feat_path)
    kan_model = safe_load_model("KAN Classifier", kan_path, {"KANLayer": KANLayer})

    if feature_extractor and kan_model:
        models_dict["CNN+KAN"] = (feature_extractor, kan_model)

    return models_dict


models_dict = load_all_models()
st.write("---")

# ==========================================
# 4️⃣ Upload and Preprocess Image (MNIST Style)
# ==========================================
uploaded_file = st.file_uploader("📤 Upload a handwritten character image", type=["png", "jpg", "jpeg"])

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

    st.image(img, caption="🖼️ Preprocessed (MNIST-style) Image", width=150)
    st.write("---")

    # ==========================================
    # 5️⃣ Prediction Functions
    # ==========================================
    def predict_cnn(model, img):
        x = img.reshape(1, 28, 28, 1)
        pred = model.predict(x, verbose=0)
        label = np.argmax(pred, axis=1)[0]
        conf = np.max(pred)
        return chr(label + 65), conf

    def predict_rnn(model, img):
        x = img.reshape(1, 28, 28)
        pred = model.predict(x, verbose=0)
        label = np.argmax(pred, axis=1)[0]
        conf = np.max(pred)
        return chr(label + 65), conf

    def predict_cnn_kan(feature_extractor, kan_model, img):
        x = img.reshape(1, 28, 28, 1)
        features = feature_extractor.predict(x, verbose=0)
        pred = kan_model.predict(features, verbose=0)
        label = np.argmax(pred, axis=1)[0]
        conf = np.max(pred)
        return chr(label + 65), conf

    # ==========================================
    # 6️⃣ Run Predictions
    # ==========================================
    st.subheader("📊 Model Predictions")
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
        st.success("✅ Predictions:")
        for name, label, conf in results:
            st.write(f"**{name}** → Predicted: `{label}` | Confidence: `{conf*100:.2f}%`")
    else:
        st.warning("⚠️ No predictions could be made. Check if models are loaded correctly.")
else:
    st.info("👆 Upload an image above to start recognition.")

# ==========================================
# 8️⃣ Footer
# ==========================================
st.markdown("---")