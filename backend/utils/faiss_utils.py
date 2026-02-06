r"""
faiss_utils_windows_utf8_fix.py (WINDOWS SPECIFIC FIX)
Khắc phục vấn đề FAISS không thể mở file UTF-8 trên Windows

VẤN ĐỀ:
FAISS C++ backend không thể xử lý đường dẫn chứa ký tự UTF-8 (tiếng Việt)
Path: D:\Dự án - python\Brain_MRI\data\faiss\brain_tumor_index.faiss
      ↑ "Dự án" = UTF-8 characters = FAISS fails

GIẢI PHÁP:
1. Copy index tới đường dẫn ngắn hơn không có UTF-8
2. Sử dụng temporary directory
3. Register path mapping
"""

import os
import json
import numpy as np
import base64
import io
import time
import shutil
from pathlib import Path
import tempfile

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("⚠️  FAISS not installed")

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


def find_project_root():
    """Find project root"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for _ in range(5):
        current_dir = os.path.dirname(current_dir)
        data_folder = os.path.join(current_dir, 'data')
        backend_folder = os.path.join(current_dir, 'backend')
        
        if os.path.isdir(data_folder) and os.path.isdir(backend_folder):
            return current_dir
    
    return os.getcwd()


def get_short_path_name(long_name):
    """
    Convert long UTF-8 paths to Windows 8.3 short format (ASCII-safe)
    
    Windows 8.3 format: No Unicode characters
    Example: "Dự án - python" becomes "DU~1" or similar ASCII format
    
    This allows FAISS C++ backend to read files with Vietnamese folder names
    """
    try:
        import ctypes
        from ctypes import wintypes
        
        # Ensure path exists first
        if not os.path.exists(long_name):
            print(f"[FAISS] ⚠️  Path doesn't exist: {long_name}")
            return None
        
        GetShortPathName = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathName.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        GetShortPathName.restype = wintypes.DWORD
        
        # Create buffer with sufficient size
        output = ctypes.create_unicode_buffer(512)
        result = GetShortPathName(long_name, output, 512)
        
        # Check if conversion succeeded (result > 0 means success)
        if result == 0:
            print(f"[FAISS] ⚠️  GetShortPathName failed with error code: {ctypes.get_last_error()}")
            return None
        
        short_path = output.value
        
        # Verify the short path is valid and different from original
        if not short_path:
            print(f"[FAISS] ⚠️  Short path is empty")
            return None
        
        # Verify it exists
        if not os.path.exists(short_path):
            print(f"[FAISS] ⚠️  Short path doesn't exist: {short_path}")
            return None
        
        # Verify it has no UTF-8 (all ASCII)
        try:
            short_path.encode('ascii')
            print(f"[FAISS] ✅ Short path is ASCII-safe: {short_path}")
            return short_path
        except UnicodeEncodeError:
            print(f"[FAISS] ⚠️  Short path still contains non-ASCII: {short_path}")
            return None
            
    except Exception as e:
        print(f"[FAISS] ⚠️  Could not get short path: {e}")
        import traceback
        traceback.print_exc()
        return None


def copy_to_temp_location(source_path):
    """
    Copy FAISS files to temp folder with ASCII-only path
    
    Temp folders typically at: C:\\Users\\USERNAME\\AppData\\Local\\Temp
    These paths contain only ASCII characters, no UTF-8
    
    This provides fallback if short path conversion fails
    """
    try:
        print("[FAISS] 📋 Copying index to ASCII-safe temporary location...")
        
        # Create temp folder with ASCII-only name
        temp_dir = tempfile.gettempdir()
        faiss_temp_dir = os.path.join(temp_dir, 'faiss_cache')
        
        if not os.path.exists(faiss_temp_dir):
            os.makedirs(faiss_temp_dir)
        
        # Copy file
        source_file = source_path
        dest_file = os.path.join(faiss_temp_dir, os.path.basename(source_file))
        
        print(f"[FAISS] Source: {source_file}")
        print(f"[FAISS] Dest:   {dest_file}")
        
        # Verify temp path is ASCII-safe
        try:
            dest_file.encode('ascii')
            print(f"[FAISS] ✅ Temp path is ASCII-safe")
        except UnicodeEncodeError:
            print(f"[FAISS] ❌ Temp path contains non-ASCII characters!")
            return None
        
        # Copy if doesn't exist or older
        copy_needed = True
        if os.path.exists(dest_file):
            source_mtime = os.path.getmtime(source_file)
            dest_mtime = os.path.getmtime(dest_file)
            if source_mtime <= dest_mtime:
                copy_needed = False
                print(f"[FAISS] ✅ Using cached copy at {dest_file}")
        
        if copy_needed:
            file_size_mb = os.path.getsize(source_file) / 1024 / 1024
            print(f"[FAISS] 📋 Copying {file_size_mb:.1f}MB...")
            shutil.copy2(source_file, dest_file)
            print(f"[FAISS] ✅ Copy complete")
        
        # Verify the copied file exists and is readable
        if not os.path.exists(dest_file):
            print(f"[FAISS] ❌ Copied file doesn't exist: {dest_file}")
            return None
        
        if not os.access(dest_file, os.R_OK):
            print(f"[FAISS] ❌ Copied file is not readable: {dest_file}")
            return None
        
        return dest_file
        
    except Exception as e:
        print(f"[FAISS] ❌ Copy failed: {e}")
        import traceback
        traceback.print_exc()
        return None


class FAISSSearcher:
    """
    FAISS Searcher with Windows UTF-8 workaround
    
    Fixes for Windows paths with UTF-8 characters:
    1. Use Windows 8.3 short path names (ASCII format)
    2. Fallback to copying to temp folder (ASCII path)
    3. Graceful error handling with simple features fallback
    """
    
    def __init__(self, index_path=None, metadata_path=None):
        """Initialize with Windows UTF-8 fix"""
        
        if not HAS_FAISS:
            raise ImportError("FAISS not installed")
        
        project_root = find_project_root()
        
        if index_path is None:
            index_path = os.path.join(project_root, 'data', 'faiss', 'brain_tumor_index.faiss')
        
        if metadata_path is None:
            metadata_path = os.path.join(project_root, 'data', 'faiss', 'case_metadata.json')
        
        self.original_index_path = os.path.abspath(index_path)
        self.original_metadata_path = os.path.abspath(metadata_path)
        
        print(f"\n[FAISS] 🔍 Windows UTF-8 Fix Initialization")
        print(f"[FAISS] Original path: {self.original_index_path}")
        
        # Check if files exist
        if not os.path.exists(self.original_index_path):
            raise FileNotFoundError(f"Index not found: {self.original_index_path}")
        
        if not os.path.exists(self.original_metadata_path):
            raise FileNotFoundError(f"Metadata not found: {self.original_metadata_path}")
        
        # Strategy 1: Try short path name first
        print(f"[FAISS] 🔧 Strategy 1: Attempting Windows short path conversion...")
        short_index_path = get_short_path_name(self.original_index_path)
        
        self.index_path = None
        
        if short_index_path and os.path.exists(short_index_path):
            print(f"[FAISS] ✅ Using short path: {short_index_path}")
            self.index_path = short_index_path
        else:
            # Strategy 2: Fallback to temp copy
            print(f"[FAISS] 🔧 Strategy 2: Short path failed, using temp copy...")
            temp_path = copy_to_temp_location(self.original_index_path)
            
            if temp_path and os.path.exists(temp_path):
                print(f"[FAISS] ✅ Using temp copy: {temp_path}")
                self.index_path = temp_path
            else:
                # Strategy 3: Last resort - try original path anyway
                print(f"[FAISS] ⚠️  All strategies failed, trying original path as last resort...")
                self.index_path = self.original_index_path
        
        self.metadata_path = self.original_metadata_path
        
        # Initialize components
        self.index = None
        self.metadata = []
        self.feature_extractor = None
        self.img_dir = None
        
        self._load_index()
        self._load_metadata()
        self._initialize_feature_extractor()
        self._find_image_directory()
        
        print(f"[FAISS] ✅ Initialized with {len(self.metadata)} cases\n")
    
    def _load_index(self):
        """Load FAISS index with error handling"""
        try:
            print(f"[FAISS] 📥 Loading index from: {self.index_path}")
            
            # Additional verification before loading
            if not os.path.exists(self.index_path):
                raise FileNotFoundError(f"Index file not found: {self.index_path}")
            
            if not os.access(self.index_path, os.R_OK):
                raise PermissionError(f"Cannot read index file: {self.index_path}")
            
            # Try to load
            self.index = faiss.read_index(self.index_path)
            print(f"[FAISS] ✅ Index loaded: {self.index.ntotal} vectors, dimension {self.index.d}")
            
        except Exception as e:
            print(f"[FAISS] ❌ Error loading index: {e}")
            
            # If we're not already using temp copy, try it as final fallback
            if self.index_path != self.original_index_path:
                print(f"[FAISS] 🔄 Trying fresh temp copy as final fallback...")
                
                # Force fresh copy
                temp_dir = tempfile.gettempdir()
                faiss_temp_dir = os.path.join(temp_dir, 'faiss_cache')
                old_cache = os.path.join(faiss_temp_dir, os.path.basename(self.original_index_path))
                
                # Remove old cache if exists
                if os.path.exists(old_cache):
                    try:
                        os.remove(old_cache)
                        print(f"[FAISS] 🗑️  Removed old cache")
                    except:
                        pass
                
                # Try fresh copy
                temp_path = copy_to_temp_location(self.original_index_path)
                if temp_path:
                    try:
                        self.index = faiss.read_index(temp_path)
                        self.index_path = temp_path
                        print(f"[FAISS] ✅ Loaded from fresh temp copy: {self.index.ntotal} vectors")
                        return
                    except Exception as e2:
                        print(f"[FAISS] ❌ Fresh temp copy also failed: {e2}")
            
            raise RuntimeError(f"Failed to load FAISS index: {e}")
    
    def _load_metadata(self):
        """Load metadata with UTF-8 encoding"""
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"[FAISS] ✅ Metadata loaded: {len(self.metadata)} cases")
        except Exception as e:
            print(f"[FAISS] ❌ Error loading metadata: {e}")
            raise
    
    def _initialize_feature_extractor(self):
        """Initialize feature extractor"""
        if not HAS_TF:
            print("[FAISS] ⚠️  TensorFlow not available")
            return
        
        try:
            import sys
            project_root = find_project_root()
            backend_path = os.path.join(project_root, 'backend')
            
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)
            
            try:
                from prediction_engine import load_model
                model = load_model()
                
                mid_layer_idx = len(model.layers) // 2
                self.feature_extractor = tf.keras.Model(
                    inputs=model.input,
                    outputs=model.layers[mid_layer_idx].output
                )
                print(f"[FAISS] ✅ Feature extractor ready")
            except Exception as e:
                print(f"[FAISS] ⚠️  Could not load model: {e}")
        except Exception as e:
            print(f"[FAISS] ⚠️  Feature extractor init failed: {e}")
    
    def _find_image_directory(self):
        """Find image directory"""
        project_root = find_project_root()
        
        possible_dirs = [
            os.path.join(project_root, 'data', 'processed', 'combined_images'),
            os.path.join(project_root, 'data', 'processed', 'mat_images'),
            os.path.join(project_root, 'data', 'images'),
        ]
        
        for img_dir in possible_dirs:
            if os.path.isdir(img_dir):
                try:
                    images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    if images:
                        self.img_dir = img_dir
                        print(f"[FAISS] ✅ Image directory: {img_dir} ({len(images)} images)")
                        return
                except:
                    pass
        
        print("[FAISS] ⚠️  Image directory not found")
    
    def extract_features(self, image) -> np.ndarray:
        """Extract features from image"""
        if self.feature_extractor is None:
            return self._extract_simple_features(image)
        
        try:
            img = image.convert('L').resize((256, 256))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = img_array[np.newaxis, :, :, np.newaxis]
            
            features = self.feature_extractor.predict(img_array, verbose=0)
            features = features.flatten().astype(np.float32)
            
            norm = np.linalg.norm(features)
            if norm > 1e-8:
                features = features / norm
            
            return features
        except Exception as e:
            print(f"[FAISS] ⚠️  Feature extraction failed: {e}")
            return self._extract_simple_features(image)
    
    def _extract_simple_features(self, image) -> np.ndarray:
        """Simple feature extraction fallback"""
        img = image.convert('L').resize((128, 128))
        img_array = np.array(img, dtype=np.float32)
        
        features = []
        for i in range(0, 256, 32):
            hist = np.sum((img_array >= i) & (img_array < i+32))
            features.append(hist)
        
        features = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features = features / norm
        
        return features
    
    def search_similar(self, query_image, k: int = 5) -> dict:
        """Search for similar cases"""
        start_time = time.time()
        
        try:
            query_features = self.extract_features(query_image)
            query_features = query_features.reshape(1, -1).astype('float32')
            
            faiss.normalize_L2(query_features)
            
            k_actual = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_features, k_actual)
            
            similar_cases = []
            for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
                idx = int(idx)
                if idx < 0 or idx >= len(self.metadata):
                    continue
                
                case_meta = self.metadata[idx]
                
                similar_cases.append({
                    "rank": rank,
                    "case_id": idx,
                    "filename": case_meta.get('filename', ''),
                    "similarity_score": float(1.0 - dist),
                    "distance": float(dist),
                    "has_tumor": case_meta.get('has_tumor', False),
                    "source": case_meta.get('source', ''),
                    "patient_id": case_meta.get('patient_id', '')
                })
            
            search_time = (time.time() - start_time) * 1000
            
            return {
                "similar_cases": similar_cases,
                "search_time_ms": round(search_time, 2),
                "total_cases": self.index.ntotal
            }
        
        except Exception as e:
            print(f"[FAISS] ❌ Search error: {e}")
            raise
    
    def get_case_info(self, case_id: int) -> dict:
        """Get case info"""
        if case_id < 0 or case_id >= len(self.metadata):
            return None
        return self.metadata[case_id]
    
    def compare_two_images(self, img1, img2) -> dict:
        """Compare two images"""
        feat1 = self.extract_features(img1)
        feat2 = self.extract_features(img2)
        
        feat1 = feat1 / (np.linalg.norm(feat1) + 1e-8)
        feat2 = feat2 / (np.linalg.norm(feat2) + 1e-8)
        
        similarity = float(np.dot(feat1, feat2))
        
        return {
            "similarity_score": round(similarity, 4),
            "distance": round(float(1.0 - similarity), 4)
        }
    
    def get_stats(self) -> dict:
        """Get statistics"""
        try:
            index_size = os.path.getsize(self.original_index_path) / (1024 * 1024)
        except:
            index_size = 0
        
        return {
            "total_cases": self.index.ntotal if self.index else 0,
            "index_size_mb": round(index_size, 2),
            "feature_dimension": self.index.d if self.index else 0,
            "is_trained": True,
            "metadata_loaded": len(self.metadata) > 0,
            "feature_extractor_ready": self.feature_extractor is not None
        }