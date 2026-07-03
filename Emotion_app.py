"""
Application Streamlit — Reconnaissance d'Émotions Faciales
Pipeline : Upload image -> Détection visage (Haar Cascade) -> Embedding (FaceNet) -> Prédiction (classifieur)

Fichiers requis dans le même dossier :
  - best_emotion_classifier.pkl
  - embedding_scaler.pkl
  - label_encoder.pkl
"""

import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image

st.set_page_config(page_title="Émotions — Analyse Faciale", page_icon="◆", layout="centered")

# ---------------------------------------------------------------------------
# Palette & identité visuelle
# ---------------------------------------------------------------------------
EMOTION_STYLE = {
    "angry":    {"color": "#E85D4E", "label": "Colère",   "icon": "▲"},
    "disgust":  {"color": "#7C9070", "label": "Dégoût",   "icon": "◆"},
    "fear":     {"color": "#6C63FF", "label": "Peur",     "icon": "●"},
    "happy":    {"color": "#F2C94C", "label": "Joie",     "icon": "★"},
    "neutral":  {"color": "#94A3B8", "label": "Neutre",   "icon": "■"},
    "sad":      {"color": "#5B8DEF", "label": "Tristesse","icon": "▼"},
    "surprise": {"color": "#F2789F", "label": "Surprise", "icon": "✦"},
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #14151F;
    color: #E8E9F3;
}

h1, h2, h3, .hero-title {
    font-family: 'Space Grotesk', sans-serif !important;
}

.hero {
    padding: 8px 0 20px 0;
    border-bottom: 1px solid #2A2C3E;
    margin-bottom: 28px;
}
.hero-eyebrow {
    color: #8B8FA8;
    font-size: 13px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.hero-title {
    font-size: 34px;
    font-weight: 700;
    color: #F5F5FA;
    margin: 0;
}
.hero-sub {
    color: #9295AD;
    font-size: 15px;
    margin-top: 8px;
    line-height: 1.5;
}

.pipeline-steps {
    display: flex;
    gap: 6px;
    margin-top: 16px;
    flex-wrap: wrap;
}
.pipeline-chip {
    background: #1E2030;
    border: 1px solid #2E3148;
    color: #B4B7D1;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
}

.result-card {
    background: linear-gradient(135deg, #1B1D2B 0%, #1F2233 100%);
    border: 1px solid #2E3148;
    border-radius: 16px;
    padding: 28px;
    margin-top: 20px;
    text-align: center;
}
.result-icon {
    font-size: 40px;
    margin-bottom: 4px;
}
.result-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 30px;
    font-weight: 700;
    margin: 4px 0;
}
.result-confidence {
    color: #9295AD;
    font-size: 14px;
    margin-top: 4px;
}

.bar-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 9px 0;
}
.bar-label {
    width: 90px;
    font-size: 13px;
    color: #C4C6DA;
    text-align: right;
    flex-shrink: 0;
}
.bar-track {
    flex-grow: 1;
    background: #21233350;
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 6px;
}
.bar-pct {
    width: 44px;
    font-size: 12px;
    color: #8B8FA8;
    flex-shrink: 0;
}

.no-face-box {
    background: #2A1B1E;
    border: 1px solid #4A2A30;
    color: #F0A5AD;
    padding: 16px 20px;
    border-radius: 12px;
    font-size: 14px;
    margin-top: 16px;
}

[data-testid="stFileUploader"] {
    border: 1px dashed #3A3D57;
    border-radius: 12px;
    padding: 6px;
    background: #191A28;
}

.footnote {
    color: #6C6F87;
    font-size: 12px;
    text-align: center;
    margin-top: 40px;
    border-top: 1px solid #2A2C3E;
    padding-top: 16px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chargement des modèles (mis en cache pour ne pas recharger à chaque interaction)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    from keras_facenet import FaceNet

    embedder = FaceNet()
    classifier = joblib.load("best_emotion_classifier.pkl")
    scaler = joblib.load("embedding_scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return embedder, classifier, scaler, label_encoder, face_cascade


embedder, classifier, scaler, label_encoder, face_cascade = load_models()
FACENET_INPUT_SIZE = (160, 160)


# ---------------------------------------------------------------------------
# Pipeline de prédiction
# ---------------------------------------------------------------------------
def detect_face(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return image_bgr[y:y + h, x:x + w], (x, y, w, h)


def get_embedding(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, FACENET_INPUT_SIZE)
    face_batch = np.expand_dims(face_resized, axis=0)
    return embedder.embeddings(face_batch)[0]


def predict_emotion(face_bgr):
    embedding = get_embedding(face_bgr).reshape(1, -1)
    embedding_scaled = scaler.transform(embedding)

    if hasattr(classifier, "predict_proba"):
        probas = classifier.predict_proba(embedding_scaled)[0]
        pred_idx = int(np.argmax(probas))
        confidence = float(probas[pred_idx])
        all_probas = dict(zip(label_encoder.classes_, probas))
    else:
        pred_idx = int(classifier.predict(embedding_scaled)[0])
        confidence = None
        all_probas = None

    return label_encoder.classes_[pred_idx], confidence, all_probas


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Projet final — Vision par ordinateur</div>
    <p class="hero-title">Analyse d'émotions faciales</p>
    <p class="hero-sub">Charge une photo avec un visage visible. Le système le détecte, calcule sa signature FaceNet, puis prédit l'émotion exprimée.</p>
    <div class="pipeline-steps">
        <span class="pipeline-chip">1 · Détection Haar Cascade</span>
        <span class="pipeline-chip">2 · Embedding FaceNet</span>
        <span class="pipeline-chip">3 · Classification SVM</span>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Déposer une image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_pil, caption="Image chargée", use_container_width=True)

    face_crop, box = detect_face(image_bgr)

    if face_crop is None:
        st.markdown(
            '<div class="no-face-box">Aucun visage détecté sur cette image. '
            'Essaie une photo avec un visage bien visible, de face et bien éclairé.</div>',
            unsafe_allow_html=True,
        )
    else:
        with col2:
            face_rgb_display = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            st.image(face_rgb_display, caption="Visage détecté", use_container_width=True)

        with st.spinner("Analyse en cours..."):
            emotion, confidence, all_probas = predict_emotion(face_crop)

        style = EMOTION_STYLE.get(emotion, {"color": "#E8E9F3", "label": emotion, "icon": "•"})

        conf_html = f'<div class="result-confidence">Confiance : {confidence:.0%}</div>' if confidence is not None else ""

        st.markdown(f"""
        <div class="result-card">
            <div class="result-icon" style="color:{style['color']}">{style['icon']}</div>
            <div class="result-label" style="color:{style['color']}">{style['label']}</div>
            {conf_html}
        </div>
        """, unsafe_allow_html=True)

        if all_probas is not None:
            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
            sorted_probas = sorted(all_probas.items(), key=lambda item: item[1], reverse=True)

            bars_html = ""
            for emo, proba in sorted_probas:
                s = EMOTION_STYLE.get(emo, {"color": "#8B8FA8", "label": emo})
                pct = proba * 100
                bars_html += f"""
                <div class="bar-row">
                    <div class="bar-label">{s['label']}</div>
                    <div class="bar-track"><div class="bar-fill" style="width:{pct}%; background:{s['color']}"></div></div>
                    <div class="bar-pct">{pct:.0f}%</div>
                </div>
                """
            st.markdown(bars_html, unsafe_allow_html=True)

st.markdown(
    '<div class="footnote">Système de Reconnaissance d\'Émotions Faciales — FaceNet + SVM</div>',
    unsafe_allow_html=True,
)