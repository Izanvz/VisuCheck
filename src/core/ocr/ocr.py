from typing import List
from PIL import Image
import pytesseract
from app.schemas import OCRSpan, BBox
from .preprocess import preprocess_for_ocr

# Si no está en PATH, descomenta y ajusta:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Config: psm 6 = blocks de texto; psm 11 = línea dispersa (pruébalos)
CONFIG_BASE = r"--oem 3 --psm 6"  # engine LSTM, auto layout
# Para fechas/lotes ayuda permitir mayúsculas, dígitos y separadores:
CONFIG_ALNUM = CONFIG_BASE + r" -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/:.-"

def ocr_spans(pil_img: Image.Image, lang: str = "eng") -> List[OCRSpan]:
    img = preprocess_for_ocr(pil_img)

    data = pytesseract.image_to_data(
        img, lang=lang, config=CONFIG_ALNUM, output_type=pytesseract.Output.DICT
    )
    spans: List[OCRSpan] = []
    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = 0.0
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        spans.append(OCRSpan(text=text, conf=conf, bbox=BBox(x=int(x), y=int(y), w=int(w), h=int(h))))
    return spans
