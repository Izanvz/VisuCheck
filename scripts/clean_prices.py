# scripts/clean_prices.py
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


def compute_iqr_bounds(values, factor=1.5):
    """Devuelve (lower, upper) usando IQR. Si no tiene sentido, devuelve (None, None)."""
    if len(values) < 4:
        return None, None

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
        return None, None

    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return lower, upper


def clean_price(val, text, lower, upper):
    """
    Aplica reglas básicas de limpieza/corrección sobre un precio individual.
    Devuelve (clean_price, status).
    """
    # 1) Rango bruto global (por seguridad)
    if val is None or val < 0.05 or val > 200.0:
        return None, "invalid_range"

    # 2) Si no tenemos bounds IQR para la imagen, simplemente lo aceptamos
    if lower is None or upper is None:
        return val, "ok"

    # 3) Si está dentro del rango IQR expandido, lo aceptamos
    if lower <= val <= upper:
        return val, "ok"

    # 4) Es un outlier. Intentamos corregir si parece un error de punto decimal.
    #    Regla simple: si val es grande pero val/10 parece razonable, probamos.
    #    Solo aplicamos corrección si el texto NO tiene letras (para evitar "GROSSE36-40", etc.).
    has_letters = any(ch.isalpha() for ch in (text or ""))

    # Probable caso: 75 -> 7.5, 78 -> 7.8, 88 -> 8.8, 60 -> 6.0
    cand_div10 = round(val / 10.0, 2)
    if not has_letters and 0.3 <= cand_div10 <= 30.0:
        # Aceptamos la corrección si el valor corregido cae en un rango razonable
        # incluso aunque siga algo fuera del IQR original (dataset pequeño).
        return cand_div10, "fixed_div10"

    # Podríamos añadir casos de /100, pero en tu pipeline ya normalizas 133 -> 1.33
    # en la fase de parsing, así que aquí no debería hacer falta.

    # 5) Si nada aplica, descartamos el precio como outlier
    return None, "outlier_discarded"


def main():
    parser = argparse.ArgumentParser(description="Limpieza avanzada de precios OCR")
    parser.add_argument(
        "--input",
        default="data/exports/ocr_detections.jsonl",
        help="JSONL de detecciones OCR (export_results)",
    )
    parser.add_argument(
        "--output",
        default="data/exports/ocr_detections_clean.jsonl",
        help="JSONL de salida con precios limpiados",
    )
    parser.add_argument(
        "--iqr_factor",
        type=float,
        default=1.5,
        help="Factor IQR para detectar outliers (por defecto 1.5)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[ERROR] No existe {in_path}")
        return

    # 1) Primera pasada: agrupar precios por imagen para calcular IQR
    prices_by_image = defaultdict(list)

    rows = list(load_jsonl(in_path))
    total_rows = len(rows)

    for row in rows:
        img = row.get("image")
        if not img:
            continue
        is_price = bool(row.get("is_price"))
        price_val = row.get("price_value", None)
        if is_price and price_val is not None:
            prices_by_image[img].append(float(price_val))

    bounds_by_image = {}
    for img, vals in prices_by_image.items():
        lower, upper = compute_iqr_bounds(vals, factor=args.iqr_factor)
        bounds_by_image[img] = (lower, upper)

    # 2) Segunda pasada: limpiar/preclasificar precios por fila
    num_prices = 0
    num_ok = 0
    num_outliers = 0
    num_fixed = 0
    num_invalid = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for row in rows:
            img = row.get("image")
            is_price = bool(row.get("is_price"))
            price_val = row.get("price_value", None)
            text = row.get("text", "")

            clean_val = None
            status = "not_price"

            if is_price and price_val is not None:
                num_prices += 1
                lower, upper = bounds_by_image.get(img, (None, None))
                clean_val, status = clean_price(float(price_val), text, lower, upper)

                if status == "ok":
                    num_ok += 1
                elif status == "fixed_div10":
                    num_fixed += 1
                elif status in ("outlier_discarded", "invalid_range"):
                    num_outliers += 1
                else:
                    num_invalid += 1
            else:
                # no es precio
                clean_val = None
                status = "not_price"

            row["clean_price"] = clean_val
            row["price_status"] = status

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] Leídas {total_rows} filas de {in_path}")
    print(f"     Detecciones marcadas como precio: {num_prices}")
    print(f"     Precios OK: {num_ok}")
    print(f"     Precios corregidos (/10): {num_fixed}")
    print(f"     Precios descartados (outlier/invalid): {num_outliers}")
    print(f"     Otros estados: {num_invalid}")
    print(f"[OK] Salida escrita en: {out_path}")


if __name__ == "__main__":
    main()
