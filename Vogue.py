import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Cargar modelo y tokenizer
tokenizer = AutoTokenizer.from_pretrained('ruta/del/modelo')
model = AutoModelForSequenceClassification.from_pretrained('ruta/del/modelo')

# Crear pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

# Streamlit App
st.title("Clasificación de Emociones en Texto")

input_text = st.text_area("Escribe un texto:")

if st.button("Predecir emoción"):
    if input_text.strip() != "":
        pred = classifier(input_text)
        st.write(f"**Emoción detectada:** {pred[0]['label']} (con {round(pred[0]['score']*100,2)}% de confianza)")
    else:
        st.warning("Por favor escribe un texto.")
