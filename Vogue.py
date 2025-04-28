import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

# Cargar modelo y tokenizer desde tus archivos locales
@st.cache_resource
def load_classifier():
    model_path = "ruta/a/tu/modelo"  # <-- CAMBIA AQUÍ: pon la ruta donde tengas config.json y pytorch_model.bin
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

# Cargar el clasificador
classifier = load_classifier()

# Streamlit App
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
