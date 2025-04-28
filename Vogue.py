import streamlit as st
from transformers import pipeline
import speech_recognition as sr

# Cargar el modelo de clasificación de emociones
@st.cache_resource
def load_classifier():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Cargar el clasificador
classifier = load_classifier()

# Función para convertir audio a texto
def audio_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Por favor, habla ahora...")
        audio = recognizer.listen(source)
        st.info("Escuchando...")
        try:
            # Convertir audio a texto usando Google Web Speech API
            text = recognizer.recognize_google(audio)
            st.success(f"Texto detectado: {text}")
            return text
        except sr.UnknownValueError:
            st.error("No se pudo entender el audio.")
            return None
        except sr.RequestError:
            st.error("Error en la solicitud al servicio de reconocimiento de voz.")
            return None

# Streamlit App
st.title("Clasificación de Emociones en Texto")

input_text = st.text_area("Escribe un texto:")

# Opción para grabar audio
if st.button("Grabar Audio"):
    input_text = audio_to_text()

if st.button("Predecir emoción"):
    if input_text and input_text.strip() != "":
        # Realizar la predicción
        pred = classifier(input_text)
        st.write(f"**Emoción detectada:** {pred[0]['label']} (con {round(pred[0]['score']*100, 2)}% de confianza)")
    else:
        st.warning("Por favor escribe o graba un texto.")
