import argparse
import json
from pathlib import Path
import cv2
import glob


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw_box(img, poly, color=(0, 255, 0), thickness=2):
    pts = [(int(x), int(y)) for x, y in poly]
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        cv2.line(img, p1, p2, color, thickness)


def visualize(json_path, img_dir, out_dir):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[SKIP] {json_path} JSON inválido: {e}")
        return

    meta = data.get("meta", {})
    image_name = meta.get("image")
    if not image_name:
        print(f"[SKIP] {json_path} no contiene meta.image (no es OCR).")
        return

    img_path = Path(img_dir) / image_name
    if not img_path.exists():
        print(f"[WARN] No encuentro la imagen original: {img_path}")
        return

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Error leyendo imagen: {img_path}")
        return

    for item in data.get("items", []):
        boxes = item.get("boxes") or item.get("box")
        if not boxes:
            continue

        # admite formato [4 puntos] o [[4 puntos], ...]
        if isinstance(boxes[0][0], (list, tuple)):
            polys = boxes
        else:
            polys = [boxes]

        for box in polys:
            if len(box) == 4:
                draw_box(img, box)

    out_path = Path(out_dir) / f"viz_{Path(json_path).stem}_all.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"[OK] Guardado: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True)
    parser.add_argument("--imgdir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import glob
    json_paths = sorted(glob.glob(args.glob))
    if not json_paths:
        print("No se encontraron JSONs")
        return

    for jp in json_paths:
        visualize(jp, args.imgdir, out_dir)



if __name__ == "__main__":
    main()
