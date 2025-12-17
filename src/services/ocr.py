# src/services/ocr.py
import paddle
from paddleocr import PaddleOCR

def _use_gpu() -> bool:
    try:
        return paddle.device.is_compiled_with_cuda() and paddle.device.get_device().startswith("gpu")
    except Exception:
        return False

def build_ocr(lang: str = "en") -> PaddleOCR:
    return PaddleOCR(
        use_gpu=_use_gpu(),
        use_angle_cls=True,
        lang=lang,              # "en" (latin) suele ir bien para DE/ES/EN; si quieres: lang="german" o "es"
        drop_score=0.5,         # filtra ruido bajo 0.5
        det_limit_side_len=1920 # por si hay imágenes grandes
    )
