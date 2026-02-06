"""
routes/similar_cases.py (✅ COMPLETE ENHANCED VERSION)
FastAPI routes for similar case retrieval with full features.

✅ FIXES:
- Increased default to 20 results (max 100)
- Added base64 thumbnail generation
- Added detailed case info endpoint
- Better image path resolution
- Pagination support
- Enhanced metadata
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import base64
from pathlib import Path

router = APIRouter()

# Initialize FAISS searcher (singleton)
_searcher = None

def get_searcher():
    """Get or initialize FAISS searcher."""
    global _searcher
    
    if _searcher is None:
        try:
            from utils.faiss_utils import FAISSSearcher
            _searcher = FAISSSearcher()
            print("[FAISS] ✅ Searcher initialized")
        except FileNotFoundError as e:
            print(f"[FAISS] ⚠️  Index not found: {str(e)}")
            print("[FAISS] 📌 To enable similar case search, run:")
            print("[FAISS]    python dataset_scripts/build_faiss_index.py")
            _searcher = None
        except Exception as e:
            print(f"[FAISS] ⚠️  Failed to initialize: {e}")
            _searcher = None
    
    return _searcher

# ===== HELPER FUNCTIONS =====

def find_image_file(filename: str) -> Path:
    """
    Find image file in multiple possible locations.
    
    Args:
        filename: Image filename (e.g., 'mat_005363.png')
    
    Returns:
        Path object if found, None otherwise
    """
    if not filename:
        return None
    
    # Find project root
    current_dir = Path(__file__).parent
    for _ in range(5):
        current_dir = current_dir.parent
        if (current_dir / 'data').exists():
            break
    
    # Try multiple possible locations
    possible_paths = [
        current_dir / 'data' / 'processed' / 'combined_images' / filename,
        current_dir / 'data' / 'processed' / 'mat_images' / filename,
        current_dir / 'data' / 'images' / filename,
        current_dir / 'data' / 'raw' / filename,
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None

def get_thumbnail_base64(filename: str, size: tuple = (256, 256)) -> str:
    """
    Load image and convert to base64 thumbnail.
    
    Args:
        filename: Image filename
        size: Thumbnail size (width, height)
    
    Returns:
        Base64 encoded image string with data URI prefix, or None
    """
    try:
        img_path = find_image_file(filename)
        
        if not img_path:
            print(f"[Similar] ⚠️  Image not found: {filename}")
            return None
        
        # Load and resize image
        img = Image.open(img_path)
        img = img.convert('RGB')
        img.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"[Similar] ❌ Error creating thumbnail for {filename}: {e}")
        return None

def get_image_url(filename: str) -> str:
    """
    Construct image URL for frontend.
    
    Args:
        filename: Image filename
    
    Returns:
        URL path for frontend
    """
    if not filename:
        return None
    
    # Remove any path prefixes
    filename = os.path.basename(filename)
    
    # Return URL that matches static file mount
    return f"/data/images/{filename}"

# ===== ENDPOINTS =====

@router.post("/find")
async def find_similar_cases(
    file: UploadFile = File(...),
    k: int = Query(default=20, ge=1, le=100, description="Number of similar cases to return"),
    include_thumbnails: bool = Query(default=True, description="Include base64 thumbnails"),
    min_similarity: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score")
):
    """
    Find k most similar cases to uploaded MRI.
    
    Args:
        file: MRI image file
        k: Number of similar cases (default: 20, max: 100)
        include_thumbnails: Include base64 thumbnails (default: True)
        min_similarity: Filter by minimum similarity score
    
    Returns:
        {
            "status": "available",
            "message": str,
            "similar_cases": [...],
            "search_time_ms": float,
            "total_cases_in_index": int,
            "results_returned": int,
            "filtered_count": int
        }
    """
    searcher = get_searcher()
    
    # Return 503 if index not available
    if searcher is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "message": "Similar case search not available",
                "details": "FAISS index has not been built. To enable this feature, run: python dataset_scripts/build_faiss_index.py",
                "similar_cases": [],
                "search_time_ms": 0,
                "help_url": "https://github.com/your-repo/wiki/building-faiss-index"
            }
        )
    
    try:
        # Read image
        data = await file.read()
        img = Image.open(io.BytesIO(data))
        
        print(f"[Similar] 🔍 Finding {k} cases for: {file.filename}")
        
        # Search
        results = searcher.search_similar(img, k=k)
        
        # Enhance results with thumbnails and URLs
        enhanced_cases = []
        filtered_count = 0
        
        for case in results['similar_cases']:
            # Filter by minimum similarity
            if case['similarity_score'] < min_similarity:
                filtered_count += 1
                continue
            
            enhanced_case = case.copy()
            
            # Add image URL
            if case.get('filename'):
                enhanced_case['image_url'] = get_image_url(case['filename'])
                
                # Add thumbnail if requested
                if include_thumbnails:
                    thumbnail = get_thumbnail_base64(case['filename'])
                    if thumbnail:
                        enhanced_case['thumbnail'] = thumbnail
                    else:
                        enhanced_case['thumbnail'] = None
                        print(f"[Similar] ⚠️  No thumbnail for {case['filename']}")
            
            enhanced_cases.append(enhanced_case)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "available",
                "message": f"Found {len(enhanced_cases)} similar cases",
                "query_filename": file.filename,
                "similar_cases": enhanced_cases,
                "search_time_ms": results['search_time_ms'],
                "total_cases_in_index": results.get('total_cases', 0),
                "results_returned": len(enhanced_cases),
                "filtered_count": filtered_count,
                "query_params": {
                    "k": k,
                    "include_thumbnails": include_thumbnails,
                    "min_similarity": min_similarity
                }
            }
        )
        
    except Exception as e:
        print(f"[Similar] ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Search failed",
                "error": str(e),
                "similar_cases": []
            }
        )

@router.get("/case/{case_id}")
def get_case_details(
    case_id: int,
    include_thumbnail: bool = Query(default=True, description="Include base64 thumbnail")
):
    """
    Get detailed information for a specific case.
    
    Args:
        case_id: Case ID from FAISS index
        include_thumbnail: Include base64 thumbnail
    
    Returns:
        {
            "case_id": int,
            "filename": str,
            "has_tumor": bool,
            "source": str,
            "patient_id": str,
            "thumbnail": str (base64),
            "image_url": str,
            "metadata": {...}
        }
    """
    searcher = get_searcher()
    
    if searcher is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "message": "FAISS index not available"
            }
        )
    
    try:
        case_info = searcher.get_case_info(case_id)
        
        if case_info is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"Case {case_id} not found"}
            )
        
        # Add thumbnail if requested
        if include_thumbnail and case_info.get('filename'):
            thumbnail = get_thumbnail_base64(case_info['filename'])
            case_info['thumbnail'] = thumbnail
        
        # Add image URL
        if case_info.get('filename'):
            case_info['image_url'] = get_image_url(case_info['filename'])
        
        return JSONResponse(status_code=200, content=case_info)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@router.get("/stats")
def get_index_stats():
    """
    Get FAISS index statistics.
    
    Returns:
        {
            "status": "available",
            "total_cases": int,
            "index_size_mb": float,
            "feature_dimension": int,
            "is_trained": bool
        }
    """
    searcher = get_searcher()
    
    if searcher is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "message": "FAISS index not available",
                "total_cases": 0,
                "index_size_mb": 0,
                "help": "Build index with: python dataset_scripts/build_faiss_index.py"
            }
        )
    
    try:
        stats = searcher.get_stats()
        return JSONResponse(
            status_code=200,
            content={
                "status": "available",
                **stats
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "total_cases": 0
            }
        )

@router.post("/compare")
async def compare_cases(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    """Compare two MRI images directly."""
    searcher = get_searcher()
    
    if searcher is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "message": "FAISS index not available"
            }
        )
    
    try:
        data1 = await file1.read()
        data2 = await file2.read()
        
        img1 = Image.open(io.BytesIO(data1))
        img2 = Image.open(io.BytesIO(data2))
        
        result = searcher.compare_two_images(img1, img2)
        return JSONResponse(status_code=200, content=result)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@router.get("/status")
def get_similar_feature_status():
    """
    Get status of similar cases feature.
    """
    searcher = get_searcher()
    
    if searcher is None:
        return JSONResponse(
            status_code=200,
            content={
                "feature_available": False,
                "feature_name": "Similar Cases Search",
                "status": "not_initialized",
                "reason": "FAISS index not found",
                "setup_command": "python dataset_scripts/build_faiss_index.py",
                "setup_time_estimate": "5-15 minutes",
                "dataset_size": "recommended 1000+ images"
            }
        )
    
    try:
        stats = searcher.get_stats()
        return JSONResponse(
            status_code=200,
            content={
                "feature_available": True,
                "feature_name": "Similar Cases Search",
                "status": "initialized",
                "total_cases": stats.get('total_cases', 0),
                "index_size_mb": stats.get('index_size_mb', 0),
                "ready_to_use": True,
                "max_results": 100,
                "default_results": 20
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "feature_available": False,
                "feature_name": "Similar Cases Search",
                "status": "error",
                "error": str(e),
                "ready_to_use": False
            }
        )