# Streamlit App
st.title("Clasificación de Emociones en Texto")

input_text = st.text_area("Escribe un texto:")

if st.button("Predecir emoción"):
    if input_text.strip() != "":
        pred = classifier(input_text)
        st.write(f"**Emoción detectada:** {pred[0]['label']} (con {round(pred[0]['score']*100,2)}% de confianza)")
    else:
        st.warning("Por favor escribe un texto.")
