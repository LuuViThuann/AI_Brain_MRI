"""
xai_explain.py
FastAPI routes for xAI (Explainable AI) endpoints.

Endpoints:
  POST /api/xai/explain - Full xAI analysis
  POST /api/xai/gradcam - Grad-CAM only
  POST /api/xai/rules - Rule-based analysis only
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import base64

# Import xAI modules
from xai.gradcam import GradCAMExplainer
from xai.rule_based import RuleBasedAnalyzer
from xai.shap_explain import SHAPExplainer, extract_features_for_shap

# Import prediction engine for model
from prediction_engine import load_model, preprocess_image

router = APIRouter()


# ===== HELPER FUNCTIONS =====

def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()
    return f"data:image/png;base64,{img_base64}"


# ===== ENDPOINTS =====

@router.post("/explain")
async def explain_full(file: UploadFile = File(...)):
    """
    Complete xAI analysis combining all methods.
    
    Returns:
        {
            "gradcam": {...},
            "rule_based": {...},
            "shap": {...},
            "combined_insights": [str]
        }
    """
    try:
        # Read image
        data = await file.read()
        img = Image.open(io.BytesIO(data))
        
        print(f"[xAI] Processing: {file.filename}")
        
        # Load model
        model = load_model()
        
        # Preprocess image
        img_array = preprocess_image(img)
        
        # Run prediction to get mask
        prediction = model.predict(img_array, verbose=0)[0, :, :, 0]
        mask = (prediction > 0.5).astype(np.float32)
        
        # 1. Grad-CAM
        print("[xAI] Running Grad-CAM...")
        gradcam_explainer = GradCAMExplainer(model)
        gradcam_result = gradcam_explainer.generate_gradcam(img_array)
        
        # Convert overlay to base64
        gradcam_result['overlay_base64'] = image_to_base64(gradcam_result['overlay'])
        del gradcam_result['overlay']  # Remove PIL object
        
        # Convert heatmap to base64
        heatmap_img = Image.fromarray((gradcam_result['heatmap'] * 255).astype(np.uint8))
        gradcam_result['heatmap_base64'] = image_to_base64(heatmap_img)
        del gradcam_result['heatmap']
        
        # 2. Rule-based analysis
        print("[xAI] Running rule-based analysis...")
        rule_analyzer = RuleBasedAnalyzer()
        mri_array = np.array(img.convert('L').resize((256, 256)))
        rule_result = rule_analyzer.analyze(mask, mri_array)
        
        # 3. SHAP explanation
        print("[xAI] Running SHAP analysis...")
        features = extract_features_for_shap(mask, mri_array)
        shap_explainer = SHAPExplainer()
        shap_result = shap_explainer.explain_prediction(features)
        
        # 4. Combined insights
        combined_insights = generate_combined_insights(
            gradcam_result, rule_result, shap_result
        )
        
        return JSONResponse(content={
            "status": "success",
            "gradcam": gradcam_result,
            "rule_based": rule_result,
            "shap": shap_result,
            "combined_insights": combined_insights
        })
        
    except Exception as e:
        print(f"[xAI] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"xAI analysis failed: {str(e)}"
        )


@router.post("/gradcam")
async def explain_gradcam(file: UploadFile = File(...)):
    """
    Grad-CAM visualization only.
    
    Returns:
        {
            "attention_score": float,
            "heatmap_base64": str,
            "overlay_base64": str,
            "focused_regions": [...]
        }
    """
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data))
        
        model = load_model()
        img_array = preprocess_image(img)
        
        explainer = GradCAMExplainer(model)
        result = explainer.generate_gradcam(img_array)
        
        # Convert to base64
        result['overlay_base64'] = image_to_base64(result['overlay'])
        heatmap_img = Image.fromarray((result['heatmap'] * 255).astype(np.uint8))
        result['heatmap_base64'] = image_to_base64(heatmap_img)
        
        del result['overlay']
        del result['heatmap']
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Grad-CAM failed: {str(e)}"
        )


@router.post("/rules")
async def explain_rules(file: UploadFile = File(...)):
    """
    Rule-based analysis only.
    
    Returns:
        {
            "tumor_area_mm2": float,
            "risk_level": str,
            "rules_triggered": [str],
            "warnings": [str]
        }
    """
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data))
        
        model = load_model()
        img_array = preprocess_image(img)
        
        # Get mask
        prediction = model.predict(img_array, verbose=0)[0, :, :, 0]
        mask = (prediction > 0.5).astype(np.float32)
        
        # Analyze
        mri_array = np.array(img.convert('L').resize((256, 256)))
        analyzer = RuleBasedAnalyzer()
        result = analyzer.analyze(mask, mri_array)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Rule-based analysis failed: {str(e)}"
        )


# ===== HELPER: COMBINED INSIGHTS =====

def generate_combined_insights(gradcam, rules, shap) -> list:
    """
    Generate combined insights from all xAI methods.
    
    Returns:
        List of insight strings
    """
    insights = []
    
    # Grad-CAM insights
    if gradcam['attention_score'] > 0.7:
        insights.append("🔍 CNN shows high confidence in identified tumor region")
    elif gradcam['attention_score'] < 0.3:
        insights.append("⚠ CNN attention is diffuse - prediction may be uncertain")
    
    # Rule-based insights
    risk = rules['risk_level']
    if risk == 'High':
        insights.append(f"⚠ High risk classification: {rules['tumor_area_mm2']}mm² tumor detected")
    elif risk == 'Low':
        insights.append(f"✓ Low risk classification: Small tumor ({rules['tumor_area_mm2']}mm²)")
    
    # Location insights
    if 'frontal' in rules['location'].lower():
        insights.append("📍 Frontal lobe location may affect motor functions")
    elif 'temporal' in rules['location'].lower():
        insights.append("📍 Temporal lobe location may affect memory/language")
    
    # SHAP insights
    top_feature = shap['top_features'][0] if shap['top_features'] else None
    if top_feature:
        importance = shap['feature_importance'][top_feature]
        insights.append(f"📊 Most important feature: {top_feature} (importance: {importance:.3f})")
    
    # Warnings from rules
    if rules['warnings']:
        insights.extend(rules['warnings'][:2])  # Add first 2 warnings
    
    return insights


# ===== INFO ENDPOINT =====

@router.get("/methods")
def get_xai_methods():
    """
    Get information about available xAI methods.
    """
    return {
        "methods": [
            {
                "name": "Grad-CAM",
                "description": "Visualizes CNN attention using gradient-weighted activation maps",
                "output": "Heatmap showing where CNN focuses",
                "speed": "Fast (~100ms)",
                "interpretability": "Medium"
            },
            {
                "name": "Rule-Based",
                "description": "Statistical analysis using medical rules and thresholds",
                "output": "Risk level, quantitative measurements",
                "speed": "Very Fast (<10ms)",
                "interpretability": "High"
            },
            {
                "name": "SHAP",
                "description": "Feature importance using Shapley values",
                "output": "Contribution of each feature to prediction",
                "speed": "Slow (~500ms)",
                "interpretability": "High"
            }
        ],
        "recommended_usage": "Use all methods together for comprehensive explanation"
    }