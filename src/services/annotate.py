from PIL import Image, ImageDraw, ImageFont
from typing import List
from app.schemas import DetectedObject, OCRSpan

def draw_annotations(pil_img: Image.Image, objects: List[DetectedObject], ocr: List[OCRSpan]) -> Image.Image:
    img = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    # tipografía opcional
    font = None
    try:
        font = ImageFont.load_default()
    except Exception:
        pass

    # Objetos (YOLO)
    for d in objects:
        x, y, w, h = d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h
        draw.rectangle([x, y, x+w, y+h], outline=(255, 0, 0), width=2)
        label = f"{d.label} {d.conf:.2f}"
        draw.text((x, max(0, y-12)), label, fill=(255, 0, 0), font=font)

    # OCR spans
    for s in ocr:
        x, y, w, h = s.bbox.x, s.bbox.y, s.bbox.w, s.bbox.h
        draw.rectangle([x, y, x+w, y+h], outline=(0, 255, 0), width=1)
        draw.text((x, y+h+2), s.text, fill=(0, 255, 0), font=font)

    return img
