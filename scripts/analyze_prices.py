import argparse
import json
from pathlib import Path
from collections import defaultdict
import math
import statistics as stats


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[SKIP] Línea JSONL inválida en {path}: {e}")
                continue


def describe_prices(values):
    """Devuelve resumen simple de una lista de precios (float)."""
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "std": None,
        }

    vals = sorted(values)
    n = len(vals)

    def percentile(p):
        if n == 1:
            return vals[0]
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return vals[int(k)]
        d0 = vals[f] * (c - k)
        d1 = vals[c] * (k - f)
        return d0 + d1

    return {
        "count": n,
        "min": vals[0],
        "max": vals[-1],
        "mean": stats.fmean(vals) if n > 0 else None,
        "median": stats.median(vals),
        "p25": percentile(25),
        "p75": percentile(75),
        "std": stats.pstdev(vals) if n > 1 else 0.0,
    }


def find_outliers(values, factor=1.5):
    """
    Devuelve índices de valores que son outliers según IQR (Q1, Q3).
    factor=1.5 es el clásico; puedes subirlo a 2.0 si quieres ser menos agresivo.
    """
    if len(values) < 4:
        return []

    vals = sorted(values)
    n = len(vals)

    def percentile(p):
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return vals[int(k)]
        d0 = vals[f] * (c - k)
        d1 = vals[c] * (k - f)
        return d0 + d1

    q1 = percentile(25)
    q3 = percentile(75)
    iqr = q3 - q1
    if iqr == 0:
        return []

    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    out_idx = []
    for idx, v in enumerate(values):
        if v < lower or v > upper:
            out_idx.append(idx)
    return out_idx


def main():
    parser = argparse.ArgumentParser(description="Análisis exploratorio de precios OCR")
    parser.add_argument(
        "--input",
        default="data/exports/ocr_detections.jsonl",
        help="Ruta al JSONL de detecciones OCR",
    )
    parser.add_argument(
        "--out",
        default="data/exports/price_summary.json",
        help="Ruta al JSON de resumen de análisis",
    )
    parser.add_argument(
        "--outliers_factor",
        type=float,
        default=1.5,
        help="Factor IQR para detectar outliers (por defecto 1.5)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] No existe {input_path}")
        return

    # Agrupar datos por imagen
    by_image = defaultdict(lambda: {
        "prices": [],
        "conf_price": [],
        "text_price": [],
    })

    total_dets = 0
    total_prices = 0

    for row in load_jsonl(input_path):
        total_dets += 1
        img = row.get("image")
        if not img:
            continue

        is_price = bool(row.get("is_price"))
        price_value = row.get("price_value", None)
        conf = float(row.get("conf", 0.0))
        text = row.get("text", "")

        if is_price and price_value is not None:
            total_prices += 1
            by_image[img]["prices"].append(float(price_value))
            by_image[img]["conf_price"].append(conf)
            by_image[img]["text_price"].append(text)

    # Construir resumen
    summary = {
        "global": {
            "total_detections": total_dets,
            "total_price_detections": total_prices,
            "images": len(by_image),
        },
        "by_image": {},
    }

    for img, info in sorted(by_image.items()):
        prices = info["prices"]
        confs = info["conf_price"]
        texts = info["text_price"]

        desc = describe_prices(prices)
        out_idx = find_outliers(prices, factor=args.outliers_factor)

        outliers = []
        for i in out_idx:
            outliers.append({
                "value": prices[i],
                "conf": confs[i] if i < len(confs) else None,
                "text": texts[i] if i < len(texts) else "",
            })

        conf_mean = stats.fmean(confs) if confs else None

        summary["by_image"][img] = {
            "prices_stats": desc,
            "num_outliers": len(outliers),
            "outliers": outliers,
            "mean_conf_price": conf_mean,
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Resumen rápido por consola
    print(f"[OK] Analizadas {summary['global']['images']} imágenes")
    print(f"     Detecciones totales: {total_dets}")
    print(f"     Detecciones de precio: {total_prices}")
    print(f"[OK] Resumen guardado en: {out_path}")

    print("\n=== Resumen por imagen (compacto) ===")
    for img, info in summary["by_image"].items():
        stats_ = info["prices_stats"]
        print(
            f"- {img}: "
            f"n={stats_['count']}, "
            f"min={stats_['min']}, "
            f"max={stats_['max']}, "
            f"mean={round(stats_['mean'],2) if stats_['mean'] is not None else None}, "
            f"outliers={info['num_outliers']}, "
            f"conf_mean={round(info['mean_conf_price'],3) if info['mean_conf_price'] is not None else None}"
        )


if __name__ == "__main__":
    main()
