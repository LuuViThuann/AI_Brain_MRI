"""
app.py (COMPLETELY FIXED VERSION)
FastAPI main application — Brain MRI Diagnosis System backend.

CRITICAL FIX:
✅ Test routes BEFORE static file mounting
✅ Absolute path resolution
✅ Detailed debugging
✅ Direct GLB file serving
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import time

# Import route modules
from routes.diagnosis import router as diagnosis_router
from routes.brain3d   import router as brain3d_router

# ===== APP INITIALIZATION =====
app = FastAPI(
    title="Brain MRI Diagnosis API",
    description="AI-powered MRI tumor detection with 3D visualization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ===== MIDDLEWARE =====

# CORS — allow frontend on any local origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log request
    status_emoji = "✅" if response.status_code < 400 else "❌"
    print(f"{status_emoji} {request.method:6} {request.url.path:40} → {response.status_code} ({process_time:.3f}s)")
    
    return response

# ===== PATH CONFIGURATION =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend", "models"))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
BRAIN_GLB_PATH = os.path.join(MODELS_DIR, "Brain.glb")

print(f"\n📁 Directory Configuration:")
print(f"   BASE_DIR:     {BASE_DIR}")
print(f"   FRONTEND_DIR: {FRONTEND_DIR}")
print(f"   MODELS_DIR:   {MODELS_DIR}")
print(f"   BRAIN_GLB:    {BRAIN_GLB_PATH}")

# ===== API ROUTES (MUST BE BEFORE STATIC FILES) =====

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Brain MRI Diagnosis API",
        "version": "1.0.0",
        "brain_model_exists": os.path.isfile(BRAIN_GLB_PATH),
        "brain_model_path": BRAIN_GLB_PATH if os.path.isfile(BRAIN_GLB_PATH) else None
    }

# ===== TEST ENDPOINTS (BEFORE STATIC MOUNTING) =====

@app.get("/test/brain-model")
async def test_brain_model():
    """Test endpoint to serve Brain.glb directly"""
    print(f"\n🔍 Testing Brain.glb access...")
    print(f"   Requested path: {BRAIN_GLB_PATH}")
    print(f"   File exists: {os.path.isfile(BRAIN_GLB_PATH)}")
    
    if os.path.isfile(BRAIN_GLB_PATH):
        file_size = os.path.getsize(BRAIN_GLB_PATH)
        print(f"   File size: {file_size / (1024*1024):.2f} MB")
        
        return FileResponse(
            path=BRAIN_GLB_PATH,
            media_type="model/gltf-binary",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Cache-Control": "public, max-age=3600",
                "Content-Type": "model/gltf-binary"
            },
            filename="Brain.glb"
        )
    else:
        # Return detailed error info
        error_info = {
            "error": "Brain.glb not found",
            "path_checked": BRAIN_GLB_PATH,
            "models_dir_exists": os.path.isdir(MODELS_DIR),
            "models_dir_path": MODELS_DIR,
            "files_in_models_dir": []
        }
        
        if os.path.isdir(MODELS_DIR):
            try:
                error_info["files_in_models_dir"] = os.listdir(MODELS_DIR)
            except Exception as e:
                error_info["list_error"] = str(e)
        
        print(f"❌ Error: {error_info}")
        return JSONResponse(content=error_info, status_code=404)

@app.get("/test/models-dir")
async def test_models_directory():
    """Test endpoint to check models directory"""
    result = {
        "models_dir": MODELS_DIR,
        "exists": os.path.isdir(MODELS_DIR),
        "files": []
    }
    
    if os.path.isdir(MODELS_DIR):
        result["files"] = os.listdir(MODELS_DIR)
    
    return result

# Mount diagnostic and 3D brain routers
app.include_router(diagnosis_router, prefix="/api", tags=["Diagnosis"])
app.include_router(brain3d_router,   prefix="/api", tags=["3D Brain"])

# ===== STATIC FILES (MUST BE LAST) =====

# Check and mount models directory
if os.path.isdir(MODELS_DIR):
    app.mount("/frontend/models", StaticFiles(directory=MODELS_DIR), name="models")
    print(f"✅ Models directory mounted")
    
    # Check Brain.glb
    if os.path.isfile(BRAIN_GLB_PATH):
        brain_size_mb = os.path.getsize(BRAIN_GLB_PATH) / (1024 * 1024)
        print(f"✅ Brain.glb found ({brain_size_mb:.2f} MB)")
        print(f"   Direct URL: http://localhost:8000/frontend/models/Brain.glb")
        print(f"   Test URL:   http://localhost:8000/test/brain-model")
    else:
        print(f"❌ Brain.glb NOT found")
        if os.path.isdir(MODELS_DIR):
            print(f"   Files in models dir: {os.listdir(MODELS_DIR)}")
else:
    print(f"❌ Models directory not found: {MODELS_DIR}")

# Mount frontend
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    print(f"✅ Frontend mounted")
else:
    print(f"❌ Frontend directory not found: {FRONTEND_DIR}")

# ===== STARTUP/SHUTDOWN EVENTS =====

@app.on_event("startup")
async def startup_event():
    """Run on server startup"""
    print("\n" + "=" * 70)
    print("  🧠 Brain MRI Diagnosis API - Server Started")
    print("=" * 70)
    print(f"\n📍 Main Endpoints:")
    print(f"   • Frontend:      http://localhost:8000")
    print(f"   • API Health:    http://localhost:8000/api/health")
    print(f"   • Diagnosis:     http://localhost:8000/api/diagnose")
    print(f"\n🧪 Debug/Test Endpoints:")
    print(f"   • Brain Model:   http://localhost:8000/test/brain-model")
    print(f"   • Models Dir:    http://localhost:8000/test/models-dir")
    print(f"   • Direct GLB:    http://localhost:8000/frontend/models/Brain.glb")
    print(f"\n📚 Documentation:")
    print(f"   • Swagger UI:    http://localhost:8000/docs")
    print(f"   • ReDoc:         http://localhost:8000/redoc")
    print("\n" + "=" * 70 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on server shutdown"""
    print("\n" + "=" * 70)
    print("  👋 Brain MRI Diagnosis API - Server Shutting Down")
    print("=" * 70 + "\n")

# ===== MAIN ENTRY POINT =====

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    
    def open_browser():
        """Open browser after server starts"""
        time.sleep(2.5)
        try:
            webbrowser.open("http://localhost:8000")
            print("🌐 Browser opened at http://localhost:8000\n")
        except Exception as e:
            print(f"⚠️  Could not open browser: {e}")
    
    # Start browser in separate thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Print startup banner
    print("\n" + "=" * 70)
    print("  🚀 Starting Brain MRI Diagnosis API Server")
    print("=" * 70)
    print("\n⏳ Initializing server...\n")
    
    # Run server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )