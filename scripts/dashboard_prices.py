# scripts/dashboard_prices.py
import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_data(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT
            image_name,
            text,
            conf,
            is_price,
            price_value,
            clean_price,
            price_status
        FROM detections
        """,
        conn,
    )
    conn.close()
    return df


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def plot_global_price_hist(df: pd.DataFrame, outdir: Path) -> None:
    clean = df[df["clean_price"].notnull()]
    if clean.empty:
        print("[WARN] No hay clean_price para el histograma global.")
        return

    plt.figure()
    clean["clean_price"].plot(kind="hist", bins=30)
    plt.title("Distribución global de precios limpios")
    plt.xlabel("Precio (€)")
    plt.ylabel("Frecuencia")
    out_path = outdir / "global_price_hist.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Histograma global de precios: {out_path}")


def plot_prices_per_image(df: pd.DataFrame, outdir: Path) -> None:
    clean = df[df["clean_price"].notnull()]
    if clean.empty:
        print("[WARN] No hay clean_price para precios por imagen.")
        return

    counts = clean.groupby("image_name")["clean_price"].count().sort_values(ascending=False)

    plt.figure(figsize=(8, 4.5))
    counts.plot(kind="bar")
    plt.title("Número de precios limpios por imagen")
    plt.xlabel("Imagen")
    plt.ylabel("Nº precios")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = outdir / "prices_per_image.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Nº de precios por imagen: {out_path}")


def plot_price_boxplot_per_image(df: pd.DataFrame, outdir: Path) -> None:
    clean = df[df["clean_price"].notnull()]
    if clean.empty:
        print("[WARN] No hay clean_price para boxplot.")
        return

    # solo imágenes con suficientes precios
    grouped = clean.groupby("image_name")
    usable = [name for name, g in grouped if len(g) >= 5]
    if not usable:
        print("[WARN] No hay imágenes con >=5 precios para boxplot.")
        return

    data = [grouped.get_group(name)["clean_price"].values for name in usable]

    plt.figure(figsize=(8, 4.5))
    plt.boxplot(data, labels=usable, showfliers=True)
    plt.title("Distribución de precios limpios por imagen")
    plt.xlabel("Imagen")
    plt.ylabel("Precio (€)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = outdir / "price_boxplot_per_image.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Boxplot de precios por imagen: {out_path}")


def plot_conf_vs_price(df: pd.DataFrame, outdir: Path) -> None:
    clean = df[(df["clean_price"].notnull()) & (df["conf"].notnull())]
    if clean.empty:
        print("[WARN] No hay datos para conf_vs_price.")
        return

    plt.figure(figsize=(6, 4.5))
    plt.scatter(clean["clean_price"], clean["conf"], s=10)
    plt.title("Confianza OCR vs precio limpio")
    plt.xlabel("Precio (€)")
    plt.ylabel("Confianza OCR")
    plt.tight_layout()
    out_path = outdir / "conf_vs_price.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Scatter conf_vs_price: {out_path}")


def save_summary_tables(df: pd.DataFrame, outdir: Path) -> None:
    clean = df[df["clean_price"].notnull()]

    # Resumen global
    global_summary = {
        "total_detections": int(len(df)),
        "total_prices_flagged": int(df["is_price"].sum()),
        "total_clean_prices": int(clean.shape[0]),
        "unique_images": int(df["image_name"].nunique()),
    }
    pd.Series(global_summary).to_json(outdir / "global_summary.json", indent=2)

    # Stats por imagen
    if not clean.empty:
        per_image = clean.groupby("image_name").agg(
            num_clean_prices=("clean_price", "count"),
            min_price=("clean_price", "min"),
            max_price=("clean_price", "max"),
            mean_price=("clean_price", "mean"),
            mean_conf=("conf", "mean"),
        )
        per_image.to_csv(outdir / "per_image_summary.csv")
        print(f"[OK] Resumen por imagen: {outdir / 'per_image_summary.csv'}")

    print(f"[OK] Resumen global: {outdir / 'global_summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="Dashboard básico de precios OCR (SQLite)")
    parser.add_argument(
        "--db",
        default="data/db/visucheck.db",
        help="Ruta a la base de datos SQLite",
    )
    parser.add_argument(
        "--outdir",
        default="data/exports/dashboard",
        help="Carpeta de salida para gráficos y resúmenes",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    if not db_path.exists():
        print(f"[ERROR] No se encuentra la base de datos: {db_path}")
        return

    print(f"[INFO] Cargando datos desde {db_path}...")
    df = load_data(db_path)
    print(f"[INFO] Detecciones totales: {len(df)}")

    # Gráficos
    plot_global_price_hist(df, outdir)
    plot_prices_per_image(df, outdir)
    plot_price_boxplot_per_image(df, outdir)
    plot_conf_vs_price(df, outdir)

    # Tablas resumen
    save_summary_tables(df, outdir)


if __name__ == "__main__":
    main()
