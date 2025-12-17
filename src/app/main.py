from fastapi import FastAPI, File, UploadFile, Form
from app.schemas import AnalyzeResponse
from services.pipeline import analyze  # <-- IMPORTA analyze (no analyze_stub)

app = FastAPI(title="VisuCheck", version="0.1")

@app.get("/")
def index():
    return {"status": "ok", "docs": "/docs", "analyze": "/analyze"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_ep(image: UploadFile = File(...), ruleset: str = Form("producto_simple_v1")):
    content = await image.read()
    result = analyze(content, ruleset)  # <-- LLAMA analyze real
    return result
