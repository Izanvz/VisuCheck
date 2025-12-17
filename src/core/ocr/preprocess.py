from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2

def to_cv(img: Image.Image):
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def from_cv(arr):
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    """
    Pipeline ligero y estable para OCR:
    - escala x2 si es pequeña
    - conversión a gris
    - normalización de contraste
    - binarización adaptativa
    - ligera apertura para limpiar ruido
    """
    img = pil_img
    # scale up si es pequeño
    if min(img.size) < 600:
        img = img.resize((img.width*2, img.height*2), Image.LANCZOS)

    cv = to_cv(img)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    # equalize / CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # adaptive threshold
    binm = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 35, 10)
    # morphological open (quita puntos)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    opened = cv2.morphologyEx(binm, cv2.MORPH_OPEN, kernel, iterations=1)
    return from_cv(cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR))
