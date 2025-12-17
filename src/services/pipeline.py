from __future__ import annotations
import os, time, math, re
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
import paddle
from paddleocr import PaddleOCR, draw_ocr
from rapidfuzz import process, fuzz

from pathlib import Path
import importlib


# ----------------------------
# Configuración y utilidades
# ----------------------------
RESULTS_DIR = Path("data/results")
TMP_DIR = Path("data/tmp")
VIZ_DIR = RESULTS_DIR / "viz"
for d in (RESULTS_DIR, TMP_DIR, VIZ_DIR):
    d.mkdir(parents=True, exist_ok=True)

BRANDS = [
    "FOLGERS", "MAXWELL HOUSE", "CAFÉ BUSTELO", "CAFE BUSTELO",
    "DR. BECKMANN", "HEITMANN", "KAVA"
]

# Acepta 1–4 dígitos, opcional símbolo €/$, con o sin decimales
_price_re = re.compile(
    r"(?<!\d)(?:\$|€)?\s*\d{1,4}(?:[.,]\d{1,2})?(?!\d)"
)



def _parse_price(raw: str) -> float | None:
    """
    Normaliza precios en formatos tipo:
    - 12,95 | 12.95 | €12,95 | $2.99
    - 1295 -> 12.95
    - 295  -> 2.95
    - 133  -> 1.33
    Aplica filtros de rango para evitar basura.
    """
    s = raw.strip()
    s = s.replace(" ", "").replace("€", "").replace("$", "")

    if not s:
        return None

    has_sep = ("," in s) or ("." in s)

    if has_sep:
        # Caso con separadores
        if "," in s and "." in s:
            # 1.234,56 -> 1234.56
            s_norm = s.replace(".", "").replace(",", ".")
        else:
            # Un solo separador -> decimal
            s_norm = s.replace(",", ".")
        try:
            val = float(s_norm)
        except ValueError:
            return None
    else:
        # Entero puro: interpretamos como céntimos si tiene 3–4 dígitos
        if not s.isdigit():
            return None
        val_int = int(s)

        if 100 <= val_int <= 9999:
            # 133 -> 1.33 ; 1295 -> 12.95
            val = val_int / 100.0
        else:
            # 1–99 -> euros enteros
            val = float(val_int)

    # Filtro de rango (ajusta si tus precios son mayores)
    if not (0.05 <= val <= 200.0):
        return None

    return round(val, 2)


def detect_shelf_bands(img: np.ndarray,
                       debug_path: Path | None = None) -> list[tuple[int, int]]:
    """
    Devuelve una lista de bandas [y0, y1] (en coords de la imagen preprocesada)
    que corresponden aproximadamente a cada balda de la estantería.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    h, w = gray.shape[:2]
    h = int(h)
    w = int(w)

    min_line_len = int(w * 0.55)
    max_gap = 25

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=120,
        minLineLength=min_line_len, maxLineGap=max_gap
    )

    ys: list[int] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            y1 = int(y1)
            y2 = int(y2)
            if abs(y1 - y2) <= 5:
                ys.append(int((y1 + y2) // 2))

    if not ys:
        return [(0, h)]

    ys = sorted(int(y) for y in ys)

    merged: list[int] = []
    cluster_radius = 18
    for y in ys:
        if not merged or abs(y - merged[-1]) > cluster_radius:
            merged.append(int(y))
        else:
            merged[-1] = int((merged[-1] + y) // 2)

    boundaries = [0] + merged + [h]
    bands: list[tuple[int, int]] = []
    min_band_height = int(h * 0.10)
    pad = 8

    for y0, y1 in zip(boundaries[:-1], boundaries[1:]):
        y0 = int(y0)
        y1 = int(y1)
        if y1 - y0 < min_band_height:
            continue
        y0p = max(0, y0 - pad)
        y1p = min(h, y1 + pad)
        bands.append((int(y0p), int(y1p)))

    if debug_path is not None:
        dbg = img.copy()
        for y in merged:
            cv2.line(dbg, (0, int(y)), (w - 1, int(y)), (0, 0, 255), 2)
        for y0, y1 in bands:
            cv2.rectangle(dbg, (0, int(y0)), (w - 1, int(y1)), (255, 0, 0), 1)
        cv2.imwrite(str(debug_path), dbg)

    return bands




def _extract_prices(texts):
    prices = []
    for t in texts:
        for m in _price_re.finditer(t):
            raw = m.group(0)
            val = _parse_price(raw)
            if val is not None:
                prices.append((raw.strip(), val))
    return prices


def _brand_fix(s: str) -> str:
    up = s.upper()
    cand, score, _ = process.extractOne(up, BRANDS, scorer=fuzz.WRatio)
    # Ajuste: aceptar “CAFE BUSTELO” como alias de “CAFÉ BUSTELO”
    if cand == "CAFE BUSTELO":
        cand = "CAFÉ BUSTELO"
    return cand if score >= 85 else s

def _bbox_center(b):
    xs = [p[0] for p in b]; ys = [p[1] for p in b]
    return (sum(xs) / 4.0, sum(ys) / 4.0)

def _bbox_height(b):
    ys = [p[1] for p in b]
    return max(ys) - min(ys)

def _bbox_area(b):
    # área aproximada por bbox mínimo envolvente
    xs = [p[0] for p in b]; ys = [p[1] for p in b]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))

def _sort_reading_order(lines):
    # lines: [ [box, (text, conf)], ... ]
    # Orden por top-left y luego por x
    def key(l):
        b = l[0]
        ys = [p[1] for p in b]; xs = [p[0] for p in b]
        return (min(ys), min(xs))
    return sorted(lines, key=key)

def _dedupe_near(texts: List[str]) -> List[str]:
    out = []
    seen = set()
    for t in texts:
        key = re.sub(r"\W+", "", t).upper()
        if key not in seen:
            out.append(t)
            seen.add(key)
    return out

# ----------------------------
# Preprocesado imagen
# ----------------------------
def preprocess(path: str) -> tuple[np.ndarray, float]:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se puede leer la imagen: {path}")

    h, w = img.shape[:2]
    long_side = max(h, w)
    scale = 1.0

    if long_side < 2600:
        scale = 2600.0 / long_side
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    # Aquí tus pasos de sharpen + CLAHE + gamma
    # ...
    # supongamos que el resultado final es 'out'
    # return out, scale

    # Si ya tienes el código de CLAHE que pusimos:
    blur = cv2.GaussianBlur(img, (0, 0), 1.2)
    sharp = cv2.addWeighted(img, 1.4, blur, -0.4, 0)

    lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    clahe_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gamma = 1.20
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    out = cv2.LUT(clahe_img, lut)

    return out, scale


# ----------------------------
# Construcción OCR
# ----------------------------
_ocr_cache: dict[str, PaddleOCR] = {}

def build_ocr(lang: str = "en", use_gpu: bool | None = None) -> PaddleOCR:
    if use_gpu is None:
        use_gpu = paddle.device.get_device().startswith("gpu")

    key = f"{lang}|{use_gpu}"
    if key in _ocr_cache:
        return _ocr_cache[key]

    ocr = PaddleOCR(
        use_gpu=use_gpu,
        lang=lang,
        ocr_version="PP-OCRv4",
        use_angle_cls=True,
        # Más resolución interna
        det_limit_side_len=4096,
        det_limit_type="max",
        # Detector DB más sensible
        det_db_thresh=0.20,
        det_db_box_thresh=0.30,
        det_db_unclip_ratio=2.2,
        # Reconocimiento menos estricto
        drop_score=0.20,
        max_text_length=80,
    )

    _ocr_cache[key] = ocr
    return ocr

def get_ocr(lang: str = "en", use_gpu: bool | None = None) -> PaddleOCR:
    return build_ocr(lang=lang, use_gpu=use_gpu)




# ----------------------------
# Fusión de tokens por línea
# ----------------------------
def _merge_line_tokens(lines: List) -> List[Tuple[str, float, list]]:
    merged = []
    cur_txt, cur_conf, cur_box = "", [], []
    last_cx, last_cy, last_h = None, None, None

    for box, (txt, conf) in lines:
        if not txt:
            continue
        cx, cy = _bbox_center(box)
        bh = _bbox_height(box)
        if bh <= 0:
            continue

        if not cur_txt:
            cur_txt, cur_conf, cur_box = txt, [float(conf)], [box]
            last_cx, last_cy, last_h = cx, cy, bh
            continue

        # Misma fila si la diferencia vertical es menor que ~0.6 de la altura media
        same_row = abs(cy - last_cy) <= 0.6 * ((bh + (last_h or bh)) / 2.0)
        # Proximidad horizontal moderada
        prox = abs(cx - last_cx) < 260

        if same_row and prox:
            cur_txt += " " + txt
            cur_conf.append(float(conf))
            cur_box.append(box)
            last_cx, last_cy, last_h = cx, cy, bh
        else:
            merged.append((cur_txt.strip(), float(np.mean(cur_conf)), cur_box))
            cur_txt, cur_conf, cur_box = txt, [float(conf)], [box]
            last_cx, last_cy, last_h = cx, cy, bh

    if cur_txt:
        merged.append((cur_txt.strip(), float(np.mean(cur_conf)), cur_box))
    return merged

# ----------------------------
# Visualización
# ----------------------------
def _resolve_font(font_path: str | None) -> str | None:
    """Devuelve una ruta TTF válida o None si no hay ninguna disponible."""
    # 1) Si el usuario pasó --font y existe, úsala
    if font_path and Path(font_path).exists():
        return font_path
    # 2) Intenta localizar simfang.ttf dentro de paddleocr
    try:
        pocr = importlib.import_module("paddleocr")
        pkg = Path(pocr.__file__).resolve().parent
        candidates = [
            pkg / "doc" / "fonts" / "simfang.ttf",
            pkg / "ppocr" / "utils" / "simfang.ttf",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
    except Exception:
        pass
    # 3) Intenta una fuente del sistema común (Windows)
    win_cand = Path("C:/Windows/Fonts/arial.ttf")
    if win_cand.exists():
        return str(win_cand)
    # 4) No hay fuente disponible
    return None

def save_viz(image_path, ocr_result, out_path: str | Path, font_path: str | None = None):
    image = cv2.imread(str(image_path))
    if image is None:
        return
    if not ocr_result or not ocr_result[0]:
        banner = image.copy()
        cv2.rectangle(banner, (0, 0), (banner.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(banner, "Sin detecciones OCR", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(out_path), banner)
        return

    boxes = [line[0] for line in ocr_result[0]]
    txts  = [line[1][0] for line in ocr_result[0]]
    scores= [line[1][1] for line in ocr_result[0]]

    font_resolved = _resolve_font(font_path)
    try:
        if font_resolved:
            pil_img = draw_ocr(image, boxes, txts, scores, font_path=font_resolved)
            pil_img.save(str(out_path))
            return
        # Si no hay fuente, seguimos con fallback sin texto
        raise RuntimeError("No TTF font available")
    except Exception:
        # Fallback: solo cajas con OpenCV
        img = image.copy()
        for b in boxes:
            xs = [p[0] for p in b]; ys = [p[1] for p in b]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(out_path), img)

def _rescale_lines(lines, scale: float):
    if not lines or scale == 1.0:
        return lines
    inv = 1.0 / scale
    new = []
    for box, (txt, conf) in lines:
        new_box = [[float(x) * inv, float(y) * inv] for (x, y) in box]
        new.append([new_box, (txt, conf)])
    return new

def _ocr_multiscale(img: np.ndarray, ocr: PaddleOCR,
                    scales=(1.0, 1.5)) -> list:
    """
    Ejecuta OCR sobre varias escalas y remapea las cajas
    a coordenadas de la imagen original (no escalada).
    Devuelve lista de [box, (txt, conf)].
    """
    h, w = img.shape[:2]
    all_lines = []

    for s in scales:
        if s == 1.0:
            scaled = img
        else:
            scaled = cv2.resize(
                img,
                (int(w * s), int(h * s)),
                interpolation=cv2.INTER_CUBIC,
            )

        res = ocr.ocr(scaled, cls=True)
        if not res or not res[0]:
            continue

        for box, (txt, conf) in res[0]:
            # Remapear coords a espacio de la imagen original (dividir por s)
            box_back = [[float(x) / s, float(y) / s] for (x, y) in box]
            all_lines.append([box_back, (txt, conf)])

    return all_lines



# ----------------------------
# Función principal de análisis
# ----------------------------
def analyze_image(image_path: str,
                  lang: str = "en",
                  font_path: str | None = None,
                  split_shelves: bool = True):
    t0 = time.time()
    img, scale = preprocess(image_path)
    h, w = img.shape[:2]

    tmp_path = TMP_DIR / "preprocessed.jpg"
    cv2.imwrite(str(tmp_path), img)
    t1 = time.time()

    ocr = build_ocr(lang=lang)

    all_lines = []
    shelf_bands: list[tuple[int, int]] = []

    if split_shelves:
        shelf_bands = detect_shelf_bands(
            img,
            debug_path=VIZ_DIR / f"bands_{Path(image_path).stem}.jpg"
        )
        for (y0, y1) in shelf_bands:
            crop = img[y0:y1, :]
            # OCR multi-escala en la banda
            lines_band = _ocr_multiscale(crop, ocr, scales=(1.0, 1.5))
            for box, (txt, conf) in lines_band:
                # Ajustar coords Y a la imagen completa (sumar y0)
                adj_box = [[x, y + y0] for (x, y) in box]
                all_lines.append([adj_box, (txt, conf)])
    else:
        # Modo global, sin bandas
        all_lines = _ocr_multiscale(img, ocr, scales=(1.0, 1.5))

    t2 = time.time()

    # Filtro por área (texto muy pequeño)
    lines = [l for l in (all_lines or []) if _bbox_area(l[0]) >= 16.0]

    # Orden + reescala a resolución original (si preprocess escaló)
    lines = _sort_reading_order(lines)
    lines = _rescale_lines(lines, scale)

    # raw_items para debug
    raw_items = []
    for box, (txt, conf) in (lines or []):
        raw_items.append({
            "text": txt,
            "conf": float(conf),
            "box": box,
        })

    merged = _merge_line_tokens(lines)

    items, texts = [], []
    for txt, conf, boxes in merged:
        fixed = _brand_fix(txt)
        items.append({"text": fixed, "conf": conf, "boxes": boxes})
        texts.append(fixed)

    texts = _dedupe_near(texts)
    prices = _extract_prices(texts)

    out = {
        "meta": {
            "image": Path(image_path).name,
            "init_ms": int((t1 - t0) * 1000),
            "ocr_ms": int((t2 - t1) * 1000),
            "total_ms": int((t2 - t0) * 1000),
            "device": "GPU" if paddle.device.get_device().startswith("gpu") else "CPU",
            "lang": lang,
            "raw_boxes": len(lines),
            "merged_lines": len(merged),
            "shelf_bands": shelf_bands,
            "scale": scale,
        },
        "items": items,
        "texts": texts,
        "prices": prices,
        "raw_items": raw_items,
    }

    try:
        save_viz(
            image_path,
            [lines],  # formato esperado por save_viz
            VIZ_DIR / f"viz_{Path(image_path).stem}.jpg",
            font_path=font_path,
        )
    except Exception as e:
        print(f"[WARN] save_viz fallo: {e}")

    return out


