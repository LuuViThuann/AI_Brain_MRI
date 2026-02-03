"""
app.py (COMPLETE INTEGRATED VERSION)
FastAPI main application — Brain MRI Diagnosis System backend.

FEATURES:
✅ Auto-open browser on startup
✅ Proper route ordering (API before static files)
✅ Request logging middleware
✅ CORS configuration
✅ Startup/shutdown events
✅ Health check endpoint
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

# ===== API ROUTES (MUST BE BEFORE STATIC FILES) =====

@app.get("/api/health")
def health_check():
    """
    Health check endpoint.
    Returns API status and version info.
    """
    return {
        "status": "ok",
        "service": "Brain MRI Diagnosis API",
        "version": "1.0.0",
        "endpoints": {
            "diagnosis": "/api/diagnose",
            "brain_3d": "/api/brain3d",
            "model_info": "/api/model-info"
        }
    }

# Mount diagnostic and 3D brain routers
app.include_router(diagnosis_router, prefix="/api", tags=["Diagnosis"])
app.include_router(brain3d_router,   prefix="/api", tags=["3D Brain"])

# ===== STATIC FILES (MUST BE LAST) =====

# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    print(f"✅ Frontend mounted: {FRONTEND_DIR}")
else:
    print(f"⚠️  Frontend directory not found: {FRONTEND_DIR}")

# ===== STARTUP/SHUTDOWN EVENTS =====

@app.on_event("startup")
async def startup_event():
    """Run on server startup"""
    print("\n" + "=" * 70)
    print("  🧠 Brain MRI Diagnosis API - Server Started")
    print("=" * 70)
    print(f"\n📍 API Endpoints:")
    print(f"   • Base URL:      http://localhost:8000/api")
    print(f"   • Health Check:  http://localhost:8000/api/health")
    print(f"   • Diagnosis:     http://localhost:8000/api/diagnose")
    print(f"   • 3D Brain:      http://localhost:8000/api/brain3d")
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
        time.sleep(2.5)  # Wait for server startup
        try:
            webbrowser.open("http://localhost:8000")
            print("🌐 Browser opened at http://localhost:8000\n")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print("   Please open manually: http://localhost:8000\n")
    
    # Start browser in separate thread (non-blocking)
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Print startup banner
    print("\n" + "=" * 70)
    print("  🚀 Starting Brain MRI Diagnosis API Server")
    print("=" * 70)
    print("\n⏳ Initializing server... (browser will open automatically)\n")
    
    # Run server
    uvicorn.run(
        app,
        host="127.0.0.1",  # localhost only (not 0.0.0.0)
        port=8000,
        log_level="info"
    )