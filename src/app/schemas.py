from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class BBox(BaseModel):
    x: int; y: int; w: int; h: int

class DetectedObject(BaseModel):
    label: str
    conf: float
    bbox: BBox

class OCRSpan(BaseModel):
    text: str
    bbox: BBox
    conf: float

class Evidence(BaseModel):
    objects: List[DetectedObject] = []
    ocr: List[OCRSpan] = []
    scores: Dict[str, float] = {}

class Failure(BaseModel):
    rule: str
    detail: str

class AnalyzeResponse(BaseModel):
    status: str = Field(..., pattern="^(valid|invalid|uncertain)$")
    failures: List[Failure] = []
    evidence: Evidence = Evidence()
    explanation: str = ""
    artifacts: Dict[str, str] = {}
