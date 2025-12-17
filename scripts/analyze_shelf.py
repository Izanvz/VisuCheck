import argparse, json, glob, sys
from pathlib import Path
from src.services.pipeline import analyze_image, RESULTS_DIR
import numpy as np

def np_encoder(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    parser = argparse.ArgumentParser(description="Analizar estanterías con VisuCheck OCR")
    parser.add_argument("--glob", default="data/samples/*.*",
                        help="Patrones separados por coma (ej. 'data/samples/*.jpg,data/samples/*.png')")
    parser.add_argument("--lang", default="en", help="Idioma OCR (en, es, etc.)")
    parser.add_argument("--outdir", default=str(RESULTS_DIR), help="Carpeta salida JSON")
    parser.add_argument("--font", default="C:\Windows\Fonts\arial.ttf", help="Ruta TTF para draw_ocr (opcional)")
    parser.add_argument("--strict", action="store_true",
                        help="Si una imagen falla, abortar (por defecto continúa).")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Admite varios patrones separados por coma
    patterns = [p.strip() for p in args.glob.split(",") if p.strip()]
    paths = sorted({p for pat in patterns for p in glob.glob(pat)})
    if not paths:
        print("No se encontraron imágenes.", file=sys.stderr)
        return

    summary, failures = [], []
    for p in paths:
        try:
            res = analyze_image(p, lang=args.lang, font_path=args.font)
            out_json = outdir / f"{Path(p).stem}.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)

            meta = res["meta"]
            summary.append({
                "image": meta["image"],
                "device": meta["device"],
                "raw_boxes": meta.get("raw_boxes", 0),
                "merged_lines": meta.get("merged_lines", 0),
                "prices_found": len(res.get("prices", [])),
                "ms_total": meta.get("total_ms", -1),
            })
            print(f"[OK] {meta['image']} | boxes={meta.get('raw_boxes',0)} "
                  f"merged={meta.get('merged_lines',0)} "
                  f"prices={len(res.get('prices',[]))} "
                  f"time={meta.get('total_ms',-1)}ms")
        except Exception as e:
            msg = f"[FAIL] {Path(p).name}: {e}"
            if args.strict:
                raise
            failures.append({"image": Path(p).name, "error": str(e)})
            print(msg, file=sys.stderr)

    # índice rápido + registro de fallos
    index_path = outdir / "_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "failures": failures}, f, default=np_encoder, ensure_ascii=False, indent=2)

    print(f"\nResumen: {len(summary)} OK, {len(failures)} fallos. Índice: {index_path}")

if __name__ == "__main__":
    main()
