import argparse
import json
import sqlite3
from pathlib import Path


def create_tables(conn):
    cur = conn.cursor()

    # Tabla de imágenes procesadas
    cur.execute("""
    CREATE TABLE IF NOT EXISTS images (
        image_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name    TEXT UNIQUE,
        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Tabla de detecciones OCR
    cur.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        det_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name    TEXT,
        text          TEXT,
        conf          REAL,
        is_price      INTEGER,
        price_value   REAL,
        clean_price   REAL,
        price_status  TEXT,

        -- Coordenadas (primer polígono)
        x1 REAL, y1 REAL,
        x2 REAL, y2 REAL,
        x3 REAL, y3 REAL,
        x4 REAL, y4 REAL,

        FOREIGN KEY(image_name) REFERENCES images(image_name)
    );
    """)

    conn.commit()


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="Importa ocr_detections_clean.jsonl a SQLite")
    parser.add_argument(
        "--input",
        default="data/exports/ocr_detections_clean.jsonl",
        help="Ruta al JSONL limpio"
    )
    parser.add_argument(
        "--db",
        default="data/db/visucheck.db",
        help="Base de datos SQLite de salida"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] No existe el JSONL: {input_path}")
        return

    conn = sqlite3.connect(db_path)
    create_tables(conn)
    cur = conn.cursor()

    inserted = 0
    images_set = set()

    for row in load_jsonl(input_path):
        image_name = row.get("image")

        if image_name not in images_set:
            cur.execute("INSERT OR IGNORE INTO images (image_name) VALUES (?)", (image_name,))
            images_set.add(image_name)

        cur.execute("""
            INSERT INTO detections (
                image_name, text, conf, is_price, price_value,
                clean_price, price_status,
                x1, y1, x2, y2, x3, y3, x4, y4
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row.get("image"),
            row.get("text"),
            float(row.get("conf", 0.0)),
            1 if row.get("is_price") else 0,
            row.get("price_value"),
            row.get("clean_price"),
            row.get("price_status"),
            row.get("x1"), row.get("y1"),
            row.get("x2"), row.get("y2"),
            row.get("x3"), row.get("y3"),
            row.get("x4"), row.get("y4"),
        ))

        inserted += 1

    conn.commit()
    conn.close()

    print(f"[OK] Importación completada.")
    print(f"     JSONL: {input_path}")
    print(f"     DB:    {db_path}")
    print(f"     Detecciones insertadas: {inserted}")
    print(f"     Imágenes registradas:   {len(images_set)}")


if __name__ == "__main__":
    main()
