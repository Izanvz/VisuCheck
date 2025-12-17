import streamlit as st
import json
from pathlib import Path
from PIL import Image

RESULTS_DIR = Path("data/results")
SAMPLES_DIR = Path("data/samples")
VIZ_DIR = Path("data/vis")

st.title("VisuCheck Dashboard")

json_files = sorted(RESULTS_DIR.glob("*.json"))

if not json_files:
    st.warning("No se han encontrado ficheros JSON en data/results")
    st.stop()

selected = st.selectbox("Selecciona un resultado", json_files)

if not selected:
    st.stop()

# 1. Cargar JSON
try:
    data = json.loads(selected.read_text(encoding="utf-8"))
except Exception as e:
    st.error(f"Error al leer el JSON: {e}")
    st.stop()

# Obtener nombre base, por ejemplo "test"
stem = selected.stem

# -------------------------------------------------------------------
# 2. Localizar imagen ORIGINAL sin boxes
# -------------------------------------------------------------------
# Aquí usamos tu JSON: data["meta"]["image"] = "test.jpg"
meta = data.get("meta", {})
image_name = meta.get("image")

original_img_path = None

if image_name:
    # Si meta.image es solo "test.jpg", la buscamos en data/samples
    candidate = SAMPLES_DIR / image_name
    if candidate.exists():
        original_img_path = candidate

# Fallback si meta.image no existe o no coincide
if original_img_path is None:
    posibles_ext = [".jpg", ".jpeg", ".png"]
    for ext in posibles_ext:
        candidate = SAMPLES_DIR / f"{stem}{ext}"
        if candidate.exists():
            original_img_path = candidate
            break

# -------------------------------------------------------------------
# 3. Localizar imagen con boxes (viz)
# -------------------------------------------------------------------
viz_img_path = None
candidatos_viz = [
    VIZ_DIR / f"viz_{stem}_all.jpg",
    VIZ_DIR / f"viz_{stem}.jpg",
    VIZ_DIR / f"{stem}_viz.jpg",
]

for c in candidatos_viz:
    if c.exists():
        viz_img_path = c
        break

# -------------------------------------------------------------------
# 4. Mostrar ambas imágenes en 2 columnas
# -------------------------------------------------------------------
col1, col2 = st.columns(2)

# Imagen original
with col1:
    st.subheader("Imagen original")
    if original_img_path and original_img_path.exists():
        st.image(Image.open(original_img_path), caption=str(original_img_path))
    else:
        st.error("No se ha encontrado la imagen original.")

# Imagen con boxes
with col2:
    st.subheader("Imagen con boxes")
    if viz_img_path and viz_img_path.exists():
        st.image(Image.open(viz_img_path), caption=str(viz_img_path))
    else:
        st.error("No se ha encontrado la imagen con boxes en data/vis.")

# -------------------------------------------------------------------
# 5. Mostrar JSON completo
# -------------------------------------------------------------------
st.subheader("Contenido del JSON")
st.json(data)
