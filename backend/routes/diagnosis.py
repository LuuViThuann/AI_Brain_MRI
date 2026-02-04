"""
routes/diagnosis.py (COMPLETE INTEGRATED VERSION)
POST /api/diagnose — Upload MRI image, get CNN prediction + Groq AI report.

FEATURES:
✅ Integrated with prediction_engine.py
✅ Integrated with groq_client.py
✅ Proper error handling
✅ File validation
✅ Detailed logging
✅ Returns full diagnosis with 3D visualization data
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import traceback
import time

# Import prediction engine and Groq client
from prediction_engine import predict_tumor
from groq_client import generate_diagnosis_report

router = APIRouter()

# ===== HELPER FUNCTIONS =====

def validate_image_file(file: UploadFile) -> None:
    """
    Validate uploaded file is a valid image.
    Raises HTTPException if invalid.
    """
    # Check content type
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/bmp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. "
                   f"Allowed types: PNG, JPG, JPEG, BMP"
        )
    
    # Check file size (max 10MB)
    if hasattr(file, 'size') and file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size: 10MB"
        )

def map_location_to_3d_key(location_hint: str) -> str:
    """
    Map CNN location hint to 3D brain location key.
    
    Examples:
        "Superior left frontal lobe" → "left_frontal"
        "Inferior right parietal lobe" → "right_parietal"
    """
    location_lower = location_hint.lower()
    
    # Determine hemisphere
    if "left" in location_lower:
        hemisphere = "left"
    elif "right" in location_lower:
        hemisphere = "right"
    else:
        hemisphere = "left"  # Default
    
    # Determine lobe
    if "frontal" in location_lower:
        lobe = "frontal"
    elif "temporal" in location_lower:
        lobe = "temporal"
    elif "parietal" in location_lower:
        lobe = "parietal"
    elif "occipital" in location_lower:
        lobe = "occipital"
    else:
        lobe = "frontal"  # Default
    
    # Determine vertical position for special cases
    if "superior" in location_lower and hemisphere == "left":
        return "superior_left"
    elif "inferior" in location_lower and hemisphere == "right":
        return "inferior_right"
    
    # Standard mapping
    return f"{hemisphere}_{lobe}"

# ===== MAIN ENDPOINT ===== 

@router.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    """
    Upload an MRI image → CNN segmentation → Groq AI report.
    
    Request:
        - file: MRI image file (PNG/JPG/JPEG/BMP)
    
    Response:
        {
            "status": "success",
            "prediction": {
                "tumor_detected": bool,
                "confidence": float,
                "tumor_area_percent": float,
                "location_hint": str,
                "location_3d_key": str,  # For 3D visualization
                "mask_shape": [256, 256]
            },
            "report": {
                "summary": str,
                "findings": str,
                "recommendation": str,
                "severity": str
            },
            "mask": [[...], ...],  # 256x256 binary segmentation mask
            "visualization": {
                "brain3d_url": str,
                "tumor_location": str,
                "tumor_size": float
            },
            "metadata": {
                "filename": str,
                "processing_time": float,
                "model_version": str
            }
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
            # If Groq fails, provide fallback report
            print(f"   ⚠️  Groq API warning: {str(e)}")
            report = {
                "summary": "Automated analysis completed",
                "findings": f"{'Tumor detected' if prediction['tumor_detected'] else 'No tumor detected'} "
                           f"with {prediction['confidence']:.1%} confidence.",
                "recommendation": "Please consult with a radiologist for professional interpretation.",
                "severity": "medium" if prediction["tumor_detected"] else "low"
            }
        
        # ===== STEP 5: MAP LOCATION FOR 3D VISUALIZATION =====
        location_3d_key = map_location_to_3d_key(prediction["location_hint"])
        
        # Calculate tumor size for 3D visualization (0.0 - 1.0 scale)
        tumor_size_3d = min(prediction["tumor_area_percent"] / 100.0 * 3, 1.0)
        
        # ===== STEP 6: BUILD COMPLETE RESPONSE =====
        processing_time = time.time() - start_time
        
        response = {
            "status": "success",
            "prediction": {
                "tumor_detected": prediction["tumor_detected"],
                "confidence": prediction["confidence"],
                "tumor_area_percent": prediction["tumor_area_percent"],
                "location_hint": prediction["location_hint"],
                "location_3d_key": location_3d_key,  # For 3D brain visualization
                "mask_shape": [256, 256]
            },
            "report": report,
            "mask": prediction["mask"],  # 256×256 binary segmentation mask
            "visualization": {
                "brain3d_url": f"/api/brain3d?location={location_3d_key}&tumor_size={tumor_size_3d:.2f}",
                "tumor_location": location_3d_key,
                "tumor_size": round(tumor_size_3d, 2)
            },
            "metadata": {
                "filename": file.filename,
                "processing_time": round(processing_time, 3),
                "model_version": "U-Net v1.0",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        print(f"   ✅ Diagnosis complete in {processing_time:.3f}s\n")
        
        return JSONResponse(content=response)
    
    # ===== ERROR HANDLING =====
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    
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
    """
    Get information about the diagnosis model.
    """
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
            "Size quantification"
        ]
    }

@router.get("/supported-formats")
def get_supported_formats():
    """
    Get list of supported image formats.
    """
    return {
        "supported_formats": ["PNG", "JPG", "JPEG", "BMP"],
        "max_file_size_mb": 10,
        "recommended_format": "PNG",
        "color_modes": ["Grayscale", "RGB"],
        "min_resolution": [128, 128],
        "recommended_resolution": [256, 256]
    }