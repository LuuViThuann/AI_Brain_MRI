"""
groq_client.py
Groq AI integration for fast diagnosis report generation.
Uses llama3-70b-8192 for medical-grade text output.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

client = Groq(api_key=GROQ_API_KEY)


def generate_diagnosis_report(
    confidence: float,
    tumor_detected: bool,
    tumor_area_percent: float,
    location_hint: str = "frontal lobe"
) -> dict:
    """
    Generate a detailed medical diagnosis report using Groq.

    Args:
        confidence: CNN prediction confidence (0–1)
        tumor_detected: Whether tumor was detected
        tumor_area_percent: Percentage of brain area occupied by tumor
        location_hint: Estimated tumor location from segmentation map

    Returns:
        dict with 'summary', 'findings', 'recommendations', 'severity'
    """

    prompt = f"""
You are an expert neuroradiology AI assistant. Based on the following
MRI analysis results, generate a structured medical report.

--- MRI Analysis Results ---
Tumor Detected: {"Yes" if tumor_detected else "No"}
Detection Confidence: {confidence * 100:.1f}%
Tumor Area Coverage: {tumor_area_percent:.2f}% of brain slice
Estimated Location: {location_hint}
Tumor Type (if detected): Low-Grade Glioma (LGG)
--- End Results ---

Generate a JSON-only response (no markdown, no extra text) with these keys:
{{
  "summary": "1-2 sentence overall summary",
  "findings": ["finding1", "finding2", "finding3"],
  "severity": "Low | Moderate | High",
  "recommendations": ["recommendation1", "recommendation2", "recommendation3"],
  "disclaimer": "Standard medical disclaimer"
}}

Be precise, professional, and clinically accurate.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a neuroradiology AI. Respond ONLY with valid JSON. "
                    "No markdown fences, no extra text."
                )
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.3
    )

    import json
    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        report = json.loads(raw)
    except json.JSONDecodeError:
        report = {
            "summary": raw[:200],
            "findings": ["Could not parse structured report."],
            "severity": "Unknown",
            "recommendations": ["Please consult a specialist."],
            "disclaimer": "This is an AI-generated report. Not a substitute for professional medical advice."
        }

    return report