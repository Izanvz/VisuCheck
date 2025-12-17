# scripts/analyze_samples.py
# ----------------------------------------------------------
# Analiza todas las imágenes de un patrón (por defecto data/samples)
# usando PaddleOCR (GPU o CPU según disponibilidad)
# y guarda los resultados en JSON (y opcionalmente TXT).
# Incluye precalentamiento (--warmup) para cargar modelos una vez.
# ----------------------------------------------------------


import argparse
import glob
import json
import os
import time
import sys
from pathlib import Path



# Añadir raíz del proyecto al sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import paddle
from src.services.pipeline import analyze_image, get_ocr


def main():
    parser = argparse.ArgumentParser(description="Analizar imágenes con VisuCheck OCR")
    parser.add_argument("--glob", default="data/samples/*.*", help="Patrón de búsqueda de imágenes")
    parser.add_argument("--lang", default="en", help="Idioma OCR (en, es, german, etc.)")
    parser.add_argument("--outdir", default="data/results", help="Carpeta donde guardar los resultados")
    parser.add_argument("--txt", action="store_true", help="Guardar también texto plano (.txt)")
    parser.add_argument("--warmup", action="store_true", help="Precargar modelo OCR antes de procesar")
    args = parser.parse_args()

    # Información básica del entorno
    device = paddle.device.get_device()
    print(f"[VisuCheck] Dispositivo actual: {device}")
    print(f"[VisuCheck] Buscando imágenes en: {args.glob}")

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print("[VisuCheck] No se encontraron imágenes.")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[VisuCheck] Guardando resultados en: {outdir}")

    # ----------------------------------------------------------
    # Warm-up opcional (descarga modelos y evita la primera latencia)
    # ----------------------------------------------------------
    if args.warmup:
        print(f"[VisuCheck] Precargando modelo OCR ({args.lang}) ...")
        t0 = time.time()
        _ = get_ocr(lang=args.lang)
        print(f"[VisuCheck] Warm-up completado en {round(time.time() - t0, 2)} s")

    # ----------------------------------------------------------
    # Procesamiento por imagen
    # ----------------------------------------------------------
    for img_path in paths:
        print(f"\n[VisuCheck] Procesando: {img_path}")
        try:
            info = analyze_image(img_path, lang=args.lang)
        except Exception as e:
            print(f"[VisuCheck] Error analizando {img_path}: {e}")
            continue

        meta = info.get("meta", {})
        print(f" - Imagen: {meta.get('image')}")
        print(f" - Tiempo total: {meta.get('total_ms')} ms")
        print(f" - Dispositivo: {meta.get('device')}")
        print(f" - Texto detectado: {len(info.get('texts', []))} líneas")
        print(f" - Precios detectados: {len(info.get('prices', []))}")

        # Guardar JSON
        out_json = outdir / f"{Path(img_path).stem}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"[VisuCheck] Guardado JSON en: {out_json}")

        # Guardar TXT (opcional)
        if args.txt:
            out_txt = outdir / f"{Path(img_path).stem}.txt"
            with open(out_txt, "w", encoding="utf-8") as f:
                for t in info.get("texts", []):
                    f.write(t + "\n")
            print(f"[VisuCheck] Guardado TXT en: {out_txt}")

    print("\n[VisuCheck] Análisis completado correctamente.")


if __name__ == "__main__":
    main()
