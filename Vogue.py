import os
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

@st.cache_resource
def load_classifier():
    # Asegúrate de usar ruta local RELATIVA
    model_path = os.path.join(os.getcwd(), "mi_modelo")  # <-- cambia "mi_modelo" por el nombre real de tu carpeta
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

classifier = load_classifier()

# Tu Streamlit App
st.title("Clasificación de Emociones en Español")

input_text = st.text_area("Escribe una frase:")

if st.button("Predecir emoción"):
    if input_text.strip() != "":
        pred = classifier(input_text, truncation=True, max_length=512)
        emotion = pred[0]['label']
        score = round(pred[0]['score'] * 100, 2)
        st.success(f"**Emoción detectada:** {emotion} ({score}% de confianza)")
    else:
        st.warning("Por favor escribe un texto para analizar.")

