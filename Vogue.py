import streamlit as st
from transformers import pipeline

# Cargar el modelo de clasificación de emociones
@st.cache_resource
def load_classifier():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Cargar el clasificador
classifier = load_classifier()

# Streamlit App
st.title("Clasificación de Emociones en Texto")

input_text = st.text_area("Escribe un texto:")

if st.button("Predecir emoción"):
    if input_text.strip() != "":
        # Realizar la predicción
        pred = classifier(input_text)
        st.write(f"**Emoción detectada:** {pred[0]['label']} (con {round(pred[0]['score']*100, 2)}% de confianza)")
    else:
        st.warning("Por favor escribe un texto.")
