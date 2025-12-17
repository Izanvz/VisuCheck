import streamlit as st
import requests
from io import BytesIO

st.title("VisuCheck — Demo MVP")
backend = st.text_input("Backend URL", "http://localhost:8000")

img = st.file_uploader("Sube imagen/PDF", type=["png","jpg","jpeg"])
ruleset = st.text_input("Ruleset", "producto_simple_v1")

if st.button("Analizar") and img:
    files = {"image": (img.name, img.getvalue(), img.type)}
    data = {"ruleset": ruleset}
    r = requests.post(f"{backend}/analyze", files=files, data=data, timeout=60)
    st.json(r.json())
