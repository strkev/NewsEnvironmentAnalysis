import streamlit as st
from src.predictor import EnvironmentNewsPredictor

st.set_page_config(page_title="Environment News Classifier")

@st.cache_resource
def get_predictor():
    return EnvironmentNewsPredictor(threshold=0.45)

predictor = get_predictor()

st.title("Environment News Classifier")
st.write("Gib einen Freitext oder Artikel ein, um die Kategorie und Modell-Konfidenz zu bestimmen.")

threshold = st.sidebar.slider("Uncertainty Threshold", min_value=0.20, max_value=0.80, value=0.45, step=0.05)
predictor.threshold = threshold

text_input = st.text_area("Artikeltext oder Überschrift:", height=150, placeholder="E.g., Solar and wind energy capacities grew significantly this year...")

if st.button("Klassifizieren"):
    if text_input.strip():
        result = predictor.predict(text_input)
        
        st.subheader("Ergebnis:")
        if result["category"] == "Uncertain / No Clear Focus":
            st.warning(f"Kategorie: **{result['category']}** (Sicherheit unter {threshold:.0%})")
        else:
            st.success(f"Kategorie: **{result['category']}** (Konfidenz: {result['confidence']:.2%})")
            
        st.write("---")
        st.write("### Wahrscheinlichkeitsverteilung aller Klassen:")
        for cat, prob in result["probabilities"].items():
            st.write(f"**{cat}**: {prob:.2%}")
            st.progress(prob)
    else:
        st.info("Bitte gib einen Text ein.")