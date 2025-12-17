import numpy as np
import cv2
import logging
import paddle
from paddleocr import PaddleOCR

def _resize_for_ocr(img, min_side=640):
    h, w = img.shape[:2]
    scale = max(1.0, min_side / min(h, w))
    if scale > 1.01:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    return img

class OCR:
    def __init__(self, lang="en", use_gpu=True, use_angle_cls=True,
                 det_model_dir=None, rec_model_dir=None, cls_model_dir=None, **extra):
        logging.getLogger("ppocr").setLevel(logging.WARNING)

        # Normaliza entradas
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.det_model_dir = det_model_dir or None
        self.rec_model_dir = rec_model_dir or None
        self.cls_model_dir = cls_model_dir or None

        # Limpia flags que pueden romper
        extra.pop("show_log", None)
        extra.pop("enable_benchmark", None)
        self.extra = extra

        self._api_mode = None  # 'device' (v3) o 'use_gpu' (v2.6)
        self._fallback_done = False

        # Decide preferencia GPU solo si la build lo soporta
        want_gpu = bool(use_gpu) and paddle.is_compiled_with_cuda()

        # Intenta API nueva (v3.x: 'device'), si falla usa API antigua (v2.6.x: 'use_gpu')
        self._build_ocr(prefer_gpu=want_gpu)

    def _build_ocr(self, prefer_gpu: bool, force_cpu: bool = False):
        # Siempre deja el device global coherente para evitar sorpresas
        try:
            paddle.set_device("gpu" if (prefer_gpu and not force_cpu) else "cpu")
        except Exception:
            paddle.set_device("cpu")
            force_cpu = True
            prefer_gpu = False

        # 1) Prueba API nueva (device=...)
        try:
            self.ocr = PaddleOCR(
                lang=self.lang,
                use_angle_cls=self.use_angle_cls,
                det_model_dir=self.det_model_dir,
                rec_model_dir=self.rec_model_dir,
                cls_model_dir=self.cls_model_dir,
                device="gpu" if (prefer_gpu and not force_cpu) else "cpu",
                **self.extra,
            )
            self._api_mode = "device"
            self.device = "gpu" if (prefer_gpu and not force_cpu) else "cpu"
            return
        except TypeError:
            # API antigua, continua abajo
            pass

        # 2) API antigua (use_gpu=...)
        self.ocr = PaddleOCR(
            lang=self.lang,
            use_angle_cls=self.use_angle_cls,
            det_model_dir=self.det_model_dir,
            rec_model_dir=self.rec_model_dir,
            cls_model_dir=self.cls_model_dir,
            use_gpu=(prefer_gpu and not force_cpu),
            **self.extra,
        )
        self._api_mode = "use_gpu"
        self.device = "gpu" if (prefer_gpu and not force_cpu) else "cpu"

    def _rebuild_cpu(self):
        # Fuerza CPU en la API que toque
        try:
            paddle.set_device("cpu")
        except Exception:
            pass
        if self._api_mode == "device":
            self.ocr = PaddleOCR(
                lang=self.lang,
                use_angle_cls=self.use_angle_cls,
                det_model_dir=self.det_model_dir,
                rec_model_dir=self.rec_model_dir,
                cls_model_dir=self.cls_model_dir,
                device="cpu",
                **self.extra,
            )
        else:  # 'use_gpu'
            self.ocr = PaddleOCR(
                lang=self.lang,
                use_angle_cls=self.use_angle_cls,
                det_model_dir=self.det_model_dir,
                rec_model_dir=self.rec_model_dir,
                cls_model_dir=self.cls_model_dir,
                use_gpu=False,
                **self.extra,
            )
        self.device = "cpu"

    def run(self, img_bgr: np.ndarray):
        img = _resize_for_ocr(img_bgr)
        try:
            return self._run_once(img)
        except RuntimeError as e:
            msg = str(e).lower()
            # Señales típicas de cuDNN/CUBLAS/DLL ausente => rehacer en CPU y reintentar 1 vez
            if (("cudnn" in msg) or ("cublas" in msg) or ("dll" in msg)) and not self._fallback_done:
                logging.warning("PaddleOCR GPU falló; rehaciendo en CPU y reintentando...")
                self._fallback_done = True
                self._rebuild_cpu()
                return self._run_once(img)
            raise

    def _run_once(self, img):
        res = self.ocr.ocr(img, cls=True)
        lines = []
        if res and res[0]:
            for (box, (txt, conf)) in res[0]:
                xys = [(float(x), float(y)) for x, y in box]
                lines.append({"poly": xys, "text": txt, "conf": float(conf)})
        return lines
