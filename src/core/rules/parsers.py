import re
from typing import Optional, Dict, Any, List

def find_price(texts: List[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    full = " ".join(texts)
    for pat in cfg["promotions"]:
        m = re.search(pat, full, flags=re.I)
        if m:
            return {"price": None, "currency": None, "promo": m.group(0), "conf": 0.8}
    for pat in cfg["price_patterns"]:
        m = re.search(pat, full, flags=re.I)
        if m:
            num = m.group(1).replace(",", ".")
            # heurística de moneda
            curr = None
            for sym in cfg.get("currency_symbols", []):
                if sym in m.group(0):
                    curr = sym; break
            return {"price": float(num), "currency": curr, "promo": None, "conf": 0.85}
    return {"price": None, "currency": None, "promo": None, "conf": 0.0}

def find_weight(texts: List[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    full = " ".join(texts)
    for pat in cfg["patterns"]:
        m = re.search(pat, full, flags=re.I)
        if m:
            return {"value": float(m.group(1)), "unit": m.group(2), "conf": 0.8}
    return {"value": None, "unit": None, "conf": 0.0}
