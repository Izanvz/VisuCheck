# scripts/export_results.py
import argparse
import json
from pathlib import Path
import glob


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="data/results/*.json",
        help="Patrón de JSONs de resultados",
    )
    parser.add_argument(
        "--out",
        default="data/exports/ocr_detections.jsonl",
        help="Fichero JSONL de salida",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(glob.glob(args.glob))
    if not json_paths:
        print("[WARN] No se encontraron JSONs")
        return

    num_rows = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for jp in json_paths:
            p = Path(jp)
            try:
                data = load_json(p)
            except Exception as e:
                print(f"[SKIP] {p} JSON inválido: {e}")
                continue

            meta = data.get("meta", {})
            image = meta.get("image")
            if not image:
                # _index.json y similares
                print(f"[SKIP] {p} sin meta.image")
                continue

            items = data.get("items", [])
            prices = data.get("prices", [])

            # mapa de textos->precios detectados (muy simple)
            price_by_text = {}
            for raw, val in prices:
                price_by_text.setdefault(raw, []).append(val)

            for det_id, item in enumerate(items):
                text = item.get("text", "")
                conf = float(item.get("conf", 0.0))
                boxes = item.get("boxes") or []
                # usamos el primer polígono como box principal
                if boxes and len(boxes[0]) == 4:
                    poly = boxes[0]
                elif boxes and isinstance(boxes[0][0], (int, float)):
                    poly = boxes
                else:
                    poly = None

                if poly is not None and len(poly) == 4:
                    x1, y1 = poly[0]
                    x2, y2 = poly[1]
                    x3, y3 = poly[2]
                    x4, y4 = poly[3]
                else:
                    x1 = y1 = x2 = y2 = x3 = y3 = x4 = y4 = None

                # heurística trivial para saber si es precio:
                # si este texto aparece literalmente como raw en prices
                is_price = False
                price_value = None
                for raw, vals in price_by_text.items():
                    if raw in text:
                        is_price = True
                        # cogemos el primero
                        price_value = vals[0]
                        break

                row = {
                    "image": image,
                    "det_id": det_id,
                    "text": text,
                    "conf": conf,
                    "is_price": is_price,
                    "price_value": price_value,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "x3": x3,
                    "y3": y3,
                    "x4": x4,
                    "y4": y4,
                }

                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                num_rows += 1

    print(f"[OK] Exportadas {num_rows} detecciones a {out_path}")


if __name__ == "__main__":
    main()
