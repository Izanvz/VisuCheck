# scripts/save_samples.py
import os, random
from datasets import load_dataset
from PIL import Image

OUT = "data/samples"
os.makedirs(OUT, exist_ok=True)

ds = load_dataset("UniDataPro/grocery-shelves", split="train")
n = len(ds)
print(f"Tamaño del split train: {n} imágenes")

# Elige 6 índices válidos sin repetición
k = min(6, n)
indices = random.sample(range(n), k)
print("Índices elegidos:", indices)

def save_rgb(img: Image.Image, path: str):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(path, format="JPEG")

for i in indices:
    img = ds[i]["image"]  # PIL.Image
    path = os.path.join(OUT, f"grocery_{i}.jpg")
    save_rgb(img, path)
    print("Guardado:", path)

print("Listo: imágenes guardadas en", OUT)
