from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re

app = FastAPI()

# ✅ Enable CORS so your HTML frontend can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # can restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReportInput(BaseModel):
    text: str

# Simple rule-based medical analyzer
NORMAL_RANGES = {
    "hemoglobin": (12.0, 15.0),
    "wbc": (4000, 11000),
    "platelets": (150000, 450000)
}

def rule_based_summary(text: str):
    text = text.lower()

    # Hemoglobin
    if "hemoglobin" in text:
        m = re.search(r"([\d.]+)", text)
        if m:
            value = float(m.group(1))
            low, high = NORMAL_RANGES["hemoglobin"]
            if value < low:
                return "Low hemoglobin, may indicate anemia."
            elif value > high:
                return "High hemoglobin, possible dehydration."
            else:
                return "Hemoglobin within normal range."

    # WBC
    if "wbc" in text:
        m = re.search(r"([\d.]+)", text)
        if m:
            value = float(m.group(1))
            low, high = NORMAL_RANGES["wbc"]
            if value < low:
                return "Low white blood cell count, possible infection or immune issue."
            elif value > high:
                return "High white blood cell count, likely infection or inflammation."
            else:
                return "WBC count within normal range."

    # Platelets
    if "platelet" in text:
        m = re.search(r"([\d.]+)", text)
        if m:
            value = float(m.group(1).replace("lakh", "00000"))
            low, high = NORMAL_RANGES["platelets"]
            if value < low:
                return "Low platelet count, increased risk of bleeding."
            elif value > high:
                return "High platelet count, may require medical review."
            else:
                return "Platelet count normal."

    return "Unable to interpret the report."

@app.post("/summarize")
async def summarize(data: ReportInput):
    summary = rule_based_summary(data.text)
    return {"summary": summary}

@app.get("/")
def root():
    return {"message": "✅ Lab2Life API is Live! Use the /summarize endpoint."}
