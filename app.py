# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import re
import os
import logging

# Optional model imports (lazy)
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import torch
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lab2life")

app = FastAPI(title="Lab2Life")

# Allow CORS (safe for dev — restrict later if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "medical_summarizer_v1"
device = torch.device("cuda" if (HAS_TRANSFORMERS and torch.cuda.is_available()) else "cpu") if HAS_TRANSFORMERS else None

# ---------------------------
# Rule-based logic
# ---------------------------
def extract_value(text, keyword):
    pattern = rf"{keyword}[^0-9\-\/]*(\-?\d+(?:\.\d+)?)"
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return float(m.group(1)) if m else None

def rule_based_summary(report: str):
    text = report.lower()
    points = []

    # Hemoglobin
    if "hemoglobin" in text:
        v = extract_value(text, "hemoglobin")
        if v is not None:
            if v < 12:
                points.append("Low hemoglobin — may indicate anemia.")
            elif v > 15:
                points.append("High hemoglobin — possible dehydration or other causes.")
            else:
                points.append("Hemoglobin is within normal range.")

    # WBC
    if "wbc" in text or "white blood" in text:
        v = extract_value(text, "wbc")
        if v is not None:
            if v < 4000:
                points.append("Low WBC — weakened immunity.")
            elif v > 11000:
                points.append("High WBC — infection or inflammation.")
            else:
                points.append("WBC count is normal.")

    # Platelets
    if "platelet" in text:
        v = extract_value(text, "platelet")
        if v is not None:
            if v < 150000:
                points.append("Low platelets — bleeding risk.")
            elif v > 450000:
                points.append("High platelets — clotting risk.")
            else:
                points.append("Platelet count is normal.")

    # RBC
    if "rbc" in text:
        v = extract_value(text, "rbc")
        if v is not None:
            if v < 4.5:
                points.append("Low RBC — anemia risk.")
            elif v > 5.9:
                points.append("High RBC — possible dehydration or disorder.")
            else:
                points.append("RBC count is normal.")

    # Sugar / glucose
    if "sugar" in text or "glucose" in text:
        v = extract_value(text, "sugar")
        if v is None:
            v = extract_value(text, "glucose")
        if v is not None:
            if v < 70:
                points.append("Low blood sugar — hypoglycemia risk.")
            elif v > 100:
                points.append("High blood sugar — possible diabetes.")
            else:
                points.append("Fasting blood sugar is normal.")

    # Creatinine
    if "creatinine" in text:
        v = extract_value(text, "creatinine")
        if v is not None:
            if v > 1.3:
                points.append("High creatinine — kidney function issue.")
            else:
                points.append("Creatinine level is normal.")

    # Blood Pressure (sys/dia)
    if "bp" in text or "blood pressure" in text:
        match = re.search(r"(\d{2,3})\s*\/\s*(\d{2,3})", text)
        if match:
            sys, dia = int(match.group(1)), int(match.group(2))
            if sys < 90 or dia < 60:
                points.append("Low BP — dizziness risk.")
            elif sys > 130 or dia > 80:
                points.append("High BP — hypertension risk.")
            else:
                points.append("Blood pressure is normal.")

    return " | ".join(points) if points else None


# ---------------------------
# Optional T5 model functions
# ---------------------------
MODEL = None
TOKENIZER = None

def load_model_if_available():
    global MODEL, TOKENIZER
    if not HAS_TRANSFORMERS:
        logger.info("transformers not installed; skipping model load")
        return False
    if os.path.isdir(MODEL_DIR):
        try:
            TOKENIZER = T5Tokenizer.from_pretrained(MODEL_DIR)
            MODEL = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
            MODEL.to(device)
            MODEL.eval()
            logger.info(f"Loaded model from {MODEL_DIR} on {device}")
            return True
        except Exception as e:
            logger.exception("Failed to load model")
            MODEL, TOKENIZER = None, None
            return False
    logger.info("No local model directory found; running rule-only mode")
    return False

# call at startup
MODEL_AVAILABLE = load_model_if_available()

def model_summarize(text: str, max_new_tokens=60, num_beams=4):
    if MODEL is None or TOKENIZER is None:
        return None
    prompt = "summarize medical report: " + text
    inputs = TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        ids = MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True
        )
    out = TOKENIZER.decode(ids[0], skip_special_tokens=True).strip()
    return out if out else None


# ---------------------------
# Hybrid decision logic
# ---------------------------
def hybrid_decide(report_text: str):
    rule = rule_based_summary(report_text)
    model = None
    if MODEL_AVAILABLE:
        try:
            model = model_summarize(report_text)
        except Exception as e:
            logger.exception("model generation failed")

    if rule and model:
        return {"summary": f"{rule} | Model: {model}", "source": "hybrid"}
    if rule:
        return {"summary": rule, "source": "rule"}
    if model:
        return {"summary": model, "source": "model"}
    return {"summary": "No medical parameters detected.", "source": "none"}


# ---------------------------
# FastAPI endpoints
# ---------------------------
class ReportIn(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(MODEL_AVAILABLE)}


@app.post("/summarize")
def summarize(report: ReportIn):
    res = hybrid_decide(report.text)
    return res


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>✅ Lab2Life API is Live!</h2>
    <p>Use the <b>/summarize</b> endpoint via POST request.</p>
    <p>Or open your frontend (index.html) locally and connect it to this link:</p>
    <code>https://lab2life.onrender.com/summarize</code>
    """


# ---------------------------
# Main entry
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
