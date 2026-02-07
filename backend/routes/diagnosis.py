"""
routes/diagnosis.py (COMPLETE INTEGRATED VERSION WITH XAI)
POST /api/diagnose — Upload MRI image, get CNN prediction + Groq AI report + XAI analysis.

FIXES APPLIED:
✅ XAI data properly integrated into diagnosis response
✅ Consistent response structure
✅ Non-blocking error handling for optional components
✅ Clear data structure documentation
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import traceback
import time
import numpy as np
import base64

# Import prediction engine and Groq client
from prediction_engine import predict_tumor, load_model, preprocess_image
from groq_client import generate_diagnosis_report

# Import XAI modules
from xai.gradcam import GradCAMExplainer
from xai.rule_based import RuleBasedAnalyzer
from xai.shap_explain import SHAPExplainer, extract_features_for_shap

router = APIRouter()

# ===== HELPER FUNCTIONS =====

def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded file is a valid image."""
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/bmp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. "
                   f"Allowed types: PNG, JPG, JPEG, BMP"
        )
    
    if hasattr(file, 'size') and file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size: 10MB"
        )

def map_location_to_3d_key(location_hint: str) -> str:
    """Map CNN location hint to 3D brain location key."""
    location_lower = location_hint.lower()
    
    if "left" in location_lower:
        hemisphere = "left"
    elif "right" in location_lower:
        hemisphere = "right"
    else:
        hemisphere = "left"
    
    if "frontal" in location_lower:
        lobe = "frontal"
    elif "temporal" in location_lower:
        lobe = "temporal"
    elif "parietal" in location_lower:
        lobe = "parietal"
    elif "occipital" in location_lower:
        lobe = "occipital"
    else:
        lobe = "frontal"
    
    if "superior" in location_lower and hemisphere == "left":
        return "superior_left"
    elif "inferior" in location_lower and hemisphere == "right":
        return "inferior_right"
    
    return f"{hemisphere}_{lobe}"

def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()
    return f"data:image/png;base64,{img_base64}"

def generate_combined_insights(gradcam, rules, shap) -> list:
    """Generate combined insights from all XAI methods."""
    insights = []
    
    # Grad-CAM insights
    if gradcam and gradcam.get('attention_score', 0) > 0.7:
        insights.append("🔍 CNN shows high confidence in identified tumor region")
    elif gradcam and gradcam.get('attention_score', 0) < 0.3:
        insights.append("⚠ CNN attention is diffuse - prediction may be uncertain")
    
    # Rule-based insights
    if rules:
        risk = rules.get('risk_level', 'Unknown')
        if risk == 'High':
            insights.append(f"⚠ High risk classification: {rules.get('tumor_area_mm2', 'unknown')}mm² tumor detected")
        elif risk == 'Low':
            insights.append(f"✓ Low risk classification: Small tumor ({rules.get('tumor_area_mm2', 'unknown')}mm²)")
        
        # Location insights
        location = rules.get('location', '').lower()
        if 'frontal' in location:
            insights.append("📍 Frontal lobe location may affect motor functions")
        elif 'temporal' in location:
            insights.append("📍 Temporal lobe location may affect memory/language")
        
        # Warnings
        warnings = rules.get('warnings', [])
        if warnings:
            insights.extend(warnings[:2])
    
    # SHAP insights
    if shap:
        top_feature = shap.get('top_features', [None])[0] if shap.get('top_features') else None
        if top_feature:
            importance = shap.get('feature_importance', {}).get(top_feature, 0)
            insights.append(f"📊 Most important feature: {top_feature} (importance: {importance:.3f})")
    
    return insights

# ===== MAIN ENDPOINT =====

@router.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    """
    Upload an MRI image → CNN segmentation → Groq AI report → XAI analysis.
    
    Response Structure:
    {
        "status": "success",
        "prediction": {
            "tumor_detected": bool,
            "confidence": float,
            "tumor_area_percent": float,
            "location_hint": str,
            "location_3d_key": str,
            "mask_shape": [int, int]
        },
        "report": {
            "summary": str,
            "findings": [str],
            "recommendations": [str],
            "severity": str,
            "disclaimer": str
        },
        "mask": [[float]],  // 256x256 binary mask
        "xai": {
            "gradcam": {
                "attention_score": float,
                "overlay_base64": str,
                "heatmap_base64": str,
                "focused_regions": [...]
            },
            "rule_based": {
                "tumor_area_mm2": float,
                "risk_level": str,
                "rules_triggered": [str],
                "warnings": [str]
            },
            "shap": {
                "top_features": [str],
                "feature_importance": {str: float},
                "shap_values": {str: float}
            },
            "combined_insights": [str]
        },
        "visualization": {...},
        "metadata": {...}
    }
    """
    start_time = time.time()
    
    try:
        # ===== STEP 1: VALIDATE FILE =====
        print(f"\n📥 Received file: {file.filename}")
        validate_image_file(file)
        
        # ===== STEP 2: READ AND LOAD IMAGE =====
        try:
            data = await file.read()
            img = Image.open(io.BytesIO(data))
            print(f"   ✅ Image loaded: {img.size} {img.mode}")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read image: {str(e)}"
            )
        
        # ===== STEP 3: CNN TUMOR PREDICTION =====
        print(f"   🔬 Running CNN tumor segmentation...")
        try:
            prediction = predict_tumor(img)
            print(f"   ✅ Prediction complete:")
            print(f"      • Tumor detected: {prediction['tumor_detected']}")
            print(f"      • Confidence: {prediction['confidence']:.2%}")
            print(f"      • Area: {prediction['tumor_area_percent']:.2f}%")
            print(f"      • Location: {prediction['location_hint']}")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"CNN prediction failed: {str(e)}"
            )
        
        # ===== STEP 4: GROQ AI REPORT GENERATION =====
        print(f"   🤖 Generating Groq AI diagnosis report...")
        try:
            report = generate_diagnosis_report(
                confidence=prediction["confidence"],
                tumor_detected=prediction["tumor_detected"],
                tumor_area_percent=prediction["tumor_area_percent"],
                location_hint=prediction["location_hint"]
            )
            print(f"   ✅ Report generated successfully")
        except Exception as e:
            print(f"   ⚠️  Groq API warning: {str(e)}")
            report = {
                "summary": "Automated analysis completed",
                "findings": f"{'Tumor detected' if prediction['tumor_detected'] else 'No tumor detected'} "
                           f"with {prediction['confidence']:.1%} confidence.",
                "recommendations": [
                    "Consult with a radiologist for professional interpretation",
                    "Consider additional imaging if needed"
                ],
                "severity": "medium" if prediction["tumor_detected"] else "low",
                "disclaimer": "⚠️ This is an AI-generated report. Not a substitute for professional medical advice."
            }
        
        # ===== STEP 5: XAI ANALYSIS (NON-BLOCKING) =====
        xai_result = {
            "error": None,
            "gradcam": None,
            "rule_based": None,
            "shap": None,
            "combined_insights": []
        }
        
        try:
            print(f"   🔍 Running XAI analysis...")
            
            model = load_model()
            img_array = preprocess_image(img)
            mask = np.array(prediction["mask"])
            mri_array = np.array(img.convert('L').resize((256, 256)))
            
            # 1. Grad-CAM
            try:
                print(f"      • Grad-CAM...")
                gradcam_explainer = GradCAMExplainer(model)
                gradcam_result = gradcam_explainer.generate_gradcam(img_array)
                
                # Convert to base64
                gradcam_result['overlay_base64'] = image_to_base64(gradcam_result['overlay'])
                del gradcam_result['overlay']
                
                heatmap_img = Image.fromarray((gradcam_result['heatmap'] * 255).astype(np.uint8))
                gradcam_result['heatmap_base64'] = image_to_base64(heatmap_img)
                del gradcam_result['heatmap']
                
                xai_result['gradcam'] = gradcam_result
                print(f"      ✅ Grad-CAM complete")
            except Exception as e:
                print(f"      ⚠️  Grad-CAM failed: {str(e)}")
            
            # 2. Rule-based analysis
            try:
                print(f"      • Rule-based...")
                rule_analyzer = RuleBasedAnalyzer()
                rule_result = rule_analyzer.analyze(mask, mri_array)
                xai_result['rule_based'] = rule_result
                
                if 'severity' not in report or not report['severity']:
                    report['severity'] = rule_result.get('risk_level', 'medium').lower()
                
                print(f"      ✅ Rule-based complete")
            except Exception as e:
                print(f"      ⚠️  Rule-based failed: {str(e)}")
            
            # 3. SHAP explanation
            try:
                print(f"      • SHAP...")
                features = extract_features_for_shap(mask, mri_array)
                shap_explainer = SHAPExplainer()
                shap_result = shap_explainer.explain_prediction(features)
                xai_result['shap'] = shap_result
                print(f"      ✅ SHAP complete")
            except Exception as e:
                print(f"      ⚠️  SHAP failed: {str(e)}")
            
            # 4. Combined insights
            combined_insights = generate_combined_insights(
                xai_result['gradcam'],
                xai_result['rule_based'],
                xai_result['shap']
            )
            xai_result['combined_insights'] = combined_insights
            
            print(f"   ✅ XAI analysis complete")
            
        except Exception as e:
            print(f"   ⚠️  XAI analysis failed: {str(e)}")
            xai_result['error'] = str(e)
        
        # ===== STEP 6: MAP LOCATION FOR 3D VISUALIZATION =====
        location_3d_key = map_location_to_3d_key(prediction["location_hint"])
        tumor_size_3d = min(prediction["tumor_area_percent"] / 100.0 * 3, 1.0)
        
        # ===== STEP 7: BUILD COMPLETE RESPONSE =====
        processing_time = time.time() - start_time
        
        detailed_metrics = None
        if xai_result.get('rule_based'):
            # Metric đã có từ rule_based analysis
            detailed_metrics = xai_result['rule_based'].get('detailed_metrics')
        
        depth_metrics = None
        if xai_result.get('rule_based'):
            depth_metrics = xai_result['rule_based'].get('depth_metrics')

        response = {
            "status": "success",
            "prediction": {
                "tumor_detected": prediction["tumor_detected"],
                "confidence": prediction["confidence"],
                "tumor_area_percent": prediction["tumor_area_percent"],
                "location_hint": prediction["location_hint"],
                "location_3d_key": location_3d_key,
                "mask_shape": [256, 256],
                "centroid_px": prediction.get("centroid_px"), 
                "centroid_normalized": prediction.get("centroid_normalized"),  
            },
            "detailed_metrics": detailed_metrics, 
            "depth_metrics": depth_metrics,
            "report": report,
            "mask": prediction["mask"],
            "xai": xai_result,
            "visualization": {
                "brain3d_url": f"/api/brain3d?location={location_3d_key}&tumor_size={tumor_size_3d:.2f}",
                "tumor_location": location_3d_key,
                "tumor_size": round(tumor_size_3d, 2)
            },
            "metadata": {
                "filename": file.filename,
                "processing_time": round(processing_time, 3),
                "model_version": "U-Net v1.0",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "xai_enabled": xai_result.get('error') is None
            }
        }
        
        print(f"   ✅ Diagnosis complete in {processing_time:.3f}s\n")
        
        return JSONResponse(content=response)
    
    # ===== ERROR HANDLING =====
    except HTTPException:
        raise
    
    except Exception as e:
        print(f"\n❌ Error in /api/diagnose:")
        print(f"   {type(e).__name__}: {str(e)}")
        print(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": str(e),
                "type": type(e).__name__
            }
        )


# ===== ADDITIONAL ENDPOINTS =====

@router.get("/model-info")
def get_model_info():
    """Get information about the diagnosis model."""
    return {
        "model_name": "U-Net Brain Tumor Segmentation",
        "version": "1.0.0",
        "architecture": "U-Net",
        "input_size": [256, 256, 1],
        "output_size": [256, 256, 1],
        "framework": "TensorFlow/Keras",
        "trained_on": "Brain MRI dataset",
        "classes": ["background", "tumor"],
        "capabilities": [
            "Tumor detection",
            "Tumor segmentation",
            "Location estimation",
            "Size quantification",
            "XAI analysis (Grad-CAM, Rules, SHAP)"
        ]
    }

@router.get("/supported-formats")
def get_supported_formats():
    """Get list of supported image formats."""
    return {
        "supported_formats": ["PNG", "JPG", "JPEG", "BMP"],
        "max_file_size_mb": 10,
        "recommended_format": "PNG",
        "color_modes": ["Grayscale", "RGB"],
        "min_resolution": [128, 128],
        "recommended_resolution": [256, 256]
    }