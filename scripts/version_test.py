from paddleocr import PaddleOCR
import paddle, time

print("Device:", paddle.device.get_device())
ocr = PaddleOCR(use_gpu=True, use_angle_cls=True, lang='en')  # lang='es' si quieres español

t0 = time.time()
res = ocr.ocr("data/samples/test.jpg", cls=True)
print("Elapsed:", round(time.time()-t0, 2), "s")

if res and res[0]:
    for box, (text, conf) in [(x[0], x[1]) for x in res[0]]:
        print(f"{conf:.3f}  {text}")
