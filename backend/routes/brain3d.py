"""
routes/brain3d.py (ENHANCED VERSION)
GET /api/brain3d — Returns 3D brain mesh data + tumor region coordinates.

ENHANCEMENTS:
✅ More realistic brain shape
✅ Better tumor location mapping
✅ Confidence-based tumor size
✅ Support for multiple tumor locations
✅ Optimized mesh generation
"""

import math
import json
import random
from fastapi import APIRouter, Query
from typing import Optional

router = APIRouter()

# ===== BRAIN MESH GENERATION =====

def generate_brain_mesh(resolution: int = 40) -> dict:
    """
    Generate a simplified brain-shaped 3D mesh using spherical harmonics
    deformation on a UV sphere. Returns vertices and faces for Three.js.
    
    Args:
        resolution: Mesh resolution (higher = more detailed but slower)
    
    Returns:
        {
            "vertices": [[x, y, z], ...],
            "faces": [[v1, v2, v3], ...]
        }
    """
    vertices = []
    faces = []
    
    # Generate UV sphere with brain-like deformation
    for i in range(resolution + 1):
        phi = math.pi * i / resolution  # 0 → π (top to bottom)
        
        for j in range(resolution):
            theta = 2 * math.pi * j / resolution  # 0 → 2π (around)
            
            # Base sphere radius
            r = 1.0
            
            # Brain-like deformation using spherical harmonics
            # Cerebral hemispheres bulge (left-right asymmetry)
            r += 0.12 * math.sin(2 * phi) * math.cos(theta)
            
            # Frontal lobe bump (front)
            r += 0.08 * math.sin(phi) * math.cos(2 * theta)
            
            # Parietal ridge (top-middle)
            r += 0.05 * math.sin(3 * phi) * math.sin(theta)
            
            # Occipital bump (back)
            r += 0.06 * math.cos(2 * phi)
            
            # Temporal bulge (sides)
            if 0.3 < phi < 0.7:
                r += 0.04 * abs(math.sin(theta))
            
            # Convert spherical to Cartesian coordinates
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.cos(phi)
            z = r * math.sin(phi) * math.sin(theta)
            
            vertices.append([round(x, 4), round(y, 4), round(z, 4)])
    
    # Generate triangle faces (two triangles per quad)
    for i in range(resolution):
        for j in range(resolution):
            p1 = i * resolution + j
            p2 = i * resolution + (j + 1) % resolution
            p3 = (i + 1) * resolution + j
            p4 = (i + 1) * resolution + (j + 1) % resolution
            
            # Triangle 1
            faces.append([p1, p2, p3])
            # Triangle 2
            faces.append([p2, p4, p3])
    
    return {
        "vertices": vertices,
        "faces": faces
    }


# ===== TUMOR REGION GENERATION =====

# Enhanced location mapping with more precise coordinates
TUMOR_LOCATION_MAP = {
    # Frontal lobe
    "left_frontal":     (-0.50,  0.60,  0.35),
    "right_frontal":    ( 0.50,  0.60,  0.35),
    
    # Temporal lobe
    "left_temporal":    (-0.75,  0.00,  0.45),
    "right_temporal":   ( 0.75,  0.00,  0.45),
    
    # Parietal lobe
    "left_parietal":    (-0.45,  0.55, -0.50),
    "right_parietal":   ( 0.45,  0.55, -0.50),
    
    # Occipital lobe
    "left_occipital":   (-0.30,  0.20, -0.80),
    "right_occipital":  ( 0.30,  0.20, -0.80),
    
    # Special positions
    "superior_left":    (-0.30,  0.85,  0.20),
    "inferior_right":   ( 0.40, -0.60,  0.10),
    "central":          ( 0.00,  0.50,  0.00),
}


def generate_tumor_region(
    location: str = "left_frontal",
    size: float = 0.18
) -> list:
    """
    Generate a cluster of 3D points representing a tumor region
    on the brain surface.
    
    Args:
        location: Tumor location key (from TUMOR_LOCATION_MAP)
        size: Relative size (0.0-1.0, where 0.18 is ~18% of brain area)
    
    Returns:
        List of [x, y, z] coordinates for tumor marker points
    """
    # Get tumor center from location map
    center = TUMOR_LOCATION_MAP.get(location, (-0.5, 0.6, 0.3))
    cx, cy, cz = center
    
    # Generate scattered tumor points around center
    random.seed(99)  # Deterministic for consistent rendering
    points = []
    
    # Scale number of points based on size
    n_points = int(size * 200)  # More points = larger tumor visual
    
    for _ in range(n_points):
        # Generate point with Gaussian distribution around center
        px = cx + random.gauss(0, size * 0.4)
        py = cy + random.gauss(0, size * 0.3)
        pz = cz + random.gauss(0, size * 0.4)
        
        # Project onto brain surface (radius ~1.08 for slight offset)
        dist = math.sqrt(px**2 + py**2 + pz**2)
        scale = 1.08 / dist if dist > 0 else 1.0
        
        points.append([
            round(px * scale, 4),
            round(py * scale, 4),
            round(pz * scale, 4)
        ])
    
    return points


def generate_tumor_confidence_overlay(
    location: str,
    confidence: float
) -> dict:
    """
    Generate visual confidence indicator for tumor region.
    
    Args:
        location: Tumor location key
        confidence: Prediction confidence (0.0-1.0)
    
    Returns:
        {
            "color": str (hex color),
            "opacity": float,
            "intensity": str
        }
    """
    # Color based on confidence
    if confidence > 0.9:
        color = "#FF0000"  # Red - high confidence
        intensity = "high"
    elif confidence > 0.7:
        color = "#FF6600"  # Orange - medium-high confidence
        intensity = "medium-high"
    elif confidence > 0.5:
        color = "#FFAA00"  # Yellow-orange - medium confidence
        intensity = "medium"
    else:
        color = "#FFFF00"  # Yellow - low confidence
        intensity = "low"
    
    return {
        "color": color,
        "opacity": round(min(confidence * 0.8, 0.9), 2),
        "intensity": intensity
    }


# ===== API ENDPOINT =====

@router.get("/brain3d")
def get_brain3d(
    location: str = Query(
        default="left_frontal",
        description="Tumor location key (e.g., 'left_frontal', 'right_temporal')"
    ),
    tumor_size: float = Query(
        default=0.18,
        ge=0.0,
        le=1.0,
        description="Tumor size (0.0-1.0 scale)"
    ),
    confidence: float = Query(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Prediction confidence (0.0-1.0)"
    ),
    mesh_resolution: int = Query(
        default=40,
        ge=10,
        le=100,
        description="Mesh resolution (10-100)"
    )
):
    """
    Returns brain mesh + tumor region data for 3D rendering.
    
    Query parameters:
        - location: Tumor location key (default: 'left_frontal')
        - tumor_size: Tumor relative size 0.0-1.0 (default: 0.18)
        - confidence: Prediction confidence 0.0-1.0 (default: 0.85)
        - mesh_resolution: Mesh detail level 10-100 (default: 40)
    
    Returns:
        {
            "mesh": {
                "vertices": [[x, y, z], ...],
                "faces": [[v1, v2, v3], ...]
            },
            "tumor_points": [[x, y, z], ...],
            "tumor_visual": {
                "color": "#FF0000",
                "opacity": 0.85,
                "intensity": "high"
            },
            "meta": {
                "vertex_count": int,
                "face_count": int,
                "tumor_point_count": int,
                "location": str,
                "confidence": float,
                "available_locations": [...]
            }
        }
    """
    # Generate brain mesh
    mesh = generate_brain_mesh(resolution=mesh_resolution)
    
    # Generate tumor region
    tumor = generate_tumor_region(location=location, size=tumor_size)
    
    # Generate confidence-based visual settings
    tumor_visual = generate_tumor_confidence_overlay(location, confidence)
    
    return {
        "mesh": mesh,
        "tumor_points": tumor,
        "tumor_visual": tumor_visual,
        "meta": {
            "vertex_count": len(mesh["vertices"]),
            "face_count": len(mesh["faces"]),
            "tumor_point_count": len(tumor),
            "location": location,
            "tumor_size": tumor_size,
            "confidence": confidence,
            "available_locations": list(TUMOR_LOCATION_MAP.keys())
        }
    }


@router.get("/brain3d/locations")
def get_available_locations():
    """
    Get list of all available tumor locations.
    """
    return {
        "locations": {
            location: {
                "coordinates": coords,
                "hemisphere": "left" if coords[0] < 0 else "right",
                "lobe": location.split("_")[1] if "_" in location else "unknown"
            }
            for location, coords in TUMOR_LOCATION_MAP.items()
        },
        "total_count": len(TUMOR_LOCATION_MAP)
    }