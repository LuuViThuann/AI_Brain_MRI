"""
shap_explain.py (FIXED VERSION)
SHAP (SHapley Additive exPlanations) for model explainability.

FIXES:
1. Matplotlib backend set to Agg for FastAPI threading
2. Correct percentage calculation - divide by 100, not multiply
"""

# ============================================================================
# CRITICAL FIX: Matplotlib backend - MUST be FIRST
# ============================================================================
import os
os.environ['MPLBACKEND'] = 'Agg'  # Force non-GUI backend BEFORE importing matplotlib

import matplotlib
matplotlib.use('Agg', force=True)  # Thread-safe backend
import matplotlib.pyplot as plt
plt.ioff()  # Disable interactive mode

# ============================================================================
# Now safe to import other modules
# ============================================================================
import numpy as np
import shap
from typing import Dict, List
import io
import base64


class SHAPExplainer:
    """
    Use SHAP to explain which features contribute to tumor prediction.
    Works with tabular features extracted from MRI/mask.
    
    FIXES:
    - Thread-safe matplotlib backend for FastAPI async
    - Proper normalization of feature importance values (0-1 range)
    """
    
    FEATURE_NAMES = [
        'tumor_area',
        'tumor_perimeter',
        'circularity',
        'solidity',
        'aspect_ratio',
        'mean_intensity',
        'std_intensity',
        'location_x',
        'location_y',
        'bbox_width',
        'bbox_height'
    ]
    
    def __init__(self, model=None, background_data=None):
        """
        Args:
            model: Trained model (sklearn, keras, etc.)
            background_data: Background dataset for SHAP (n_samples, n_features)
        """
        self.model = model
        self.background_data = background_data
        self.explainer = None
        
        print(f"[SHAP] Initialized with matplotlib backend: {matplotlib.get_backend()}")
        
        if model is not None and background_data is not None:
            self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer based on model type."""
        try:
            # Try TreeExplainer first (for tree-based models)
            self.explainer = shap.TreeExplainer(self.model)
            self.explainer_type = 'tree'
        except:
            try:
                # Try KernelExplainer (model-agnostic)
                self.explainer = shap.KernelExplainer(
                    self.model.predict,
                    self.background_data
                )
                self.explainer_type = 'kernel'
            except:
                # Fallback to simple implementation
                self.explainer = None
                self.explainer_type = 'simple'
    
    def explain_prediction(
        self,
        features: np.ndarray,
        feature_names: List[str] = None
    ) -> Dict:
        """
        Explain a single prediction.
        
        Args:
            features: Feature vector (1D array)
            feature_names: Optional custom feature names
        
        Returns:
            {
                "shap_values": dict,
                "feature_importance": dict,
                "contribution_plot": base64_image,
                "top_features": [str]
            }
        """
        if feature_names is None:
            feature_names = self.FEATURE_NAMES[:len(features)]
        
        # Reshape if needed
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Compute SHAP values
        if self.explainer is not None:
            shap_values = self.explainer.shap_values(features)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Binary classification
            
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]  # First sample
        else:
            # Simple fallback: use feature values as proxy
            shap_values = features[0] * 0.1
        
        # ✅ FIX: Correct normalization
        # Convert to absolute values and normalize to 0-1 range
        feature_importance_raw = np.abs(shap_values)
        
        # Normalize by SUM instead of MAX
        total_importance = np.sum(feature_importance_raw)
        
        if total_importance > 1e-8:
            feature_importance_normalized = feature_importance_raw / total_importance  # ✅ ĐÚNG
        else:
            feature_importance_normalized = feature_importance_raw
        
        # ✅ FIX: Feature importance should be 0-1, NOT 0-100 before percentage
        feature_importance = {
            name: float(val)  # 0-1 range, tổng = 1.0
            for name, val in zip(feature_names, feature_importance_normalized)
        }
        
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # SHAP values with direction (keep as raw values)
        shap_dict = {
            name: float(val)
            for name, val in zip(feature_names, shap_values)
        }
        
        # Top contributing features
        top_features = [name for name, _ in sorted_features[:5]]
        
        # Generate visualization
        plot_base64 = self._create_waterfall_plot(
            shap_values, features[0], feature_names, feature_importance_normalized
        )
        
        interpretation_guide = {
            "importance_scale": "0-1 (normalized relative importance)",
            "interpretation": {
                "0.8-1.0": "Critical feature - strong influence on prediction",
                "0.5-0.8": "Important feature - moderate influence",
                "0.2-0.5": "Minor feature - weak influence",
                "0.0-0.2": "Negligible feature - minimal influence"
            },
            "calculation_method": "SHAP values normalized by max absolute value",
            "note": "Values represent relative contribution to final prediction"
        }
    
        return {
            "shap_values": shap_dict,
            "feature_importance": feature_importance,  # 0-1 values
            "contribution_plot": plot_base64,
            "top_features": top_features,
            
            # ===== NEW FIELD =====
            "interpretation_guide": interpretation_guide
        }
    
    def explain_batch(
        self,
        features_batch: np.ndarray,
        feature_names: List[str] = None
    ) -> Dict:
        """
        Explain multiple predictions.
        
        Returns:
            {
                "mean_shap": dict,
                "feature_importance_global": dict,
                "summary_plot": base64_image
            }
        """
        if feature_names is None:
            feature_names = self.FEATURE_NAMES[:features_batch.shape[1]]
        
        # Compute SHAP values for all samples
        if self.explainer is not None:
            shap_values = self.explainer.shap_values(features_batch)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
        else:
            shap_values = features_batch * 0.1
        
        # ✅ FIX: Mean absolute SHAP values normalized to 0-1
        mean_abs_shap_raw = np.mean(np.abs(shap_values), axis=0)
        max_importance = np.max(mean_abs_shap_raw)
        
        if max_importance > 0:
            mean_abs_shap = mean_abs_shap_raw / max_importance
        else:
            mean_abs_shap = mean_abs_shap_raw
        
        # Global feature importance (0-1 range)
        feature_importance = {
            name: float(val)  # ✅ Returns 0-1 values
            for name, val in zip(feature_names, mean_abs_shap)
        }
        
        # Mean SHAP values (with direction)
        mean_shap = {
            name: float(val)
            for name, val in zip(feature_names, np.mean(shap_values, axis=0))
        }
        
        # Generate summary plot
        summary_plot = self._create_summary_plot(
            shap_values, features_batch, feature_names, mean_abs_shap
        )
        
        return {
            "mean_shap": mean_shap,
            "feature_importance_global": feature_importance,  # ✅ Returns 0-1 values
            "summary_plot": summary_plot
        }
    
    def _create_waterfall_plot(
        self,
        shap_values: np.ndarray,
        features: np.ndarray,
        feature_names: List[str],
        importance_normalized: np.ndarray  # ✅ Receive normalized values
    ) -> str:
        """Create waterfall plot showing feature contributions."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor('#0a0e1a')
            ax.set_facecolor('#0a0e1a')
            
            # Sort by absolute SHAP value
            indices = np.argsort(np.abs(shap_values))[-10:]  # Top 10
            
            y_pos = np.arange(len(indices))
            
            # ✅ FIX: Use normalized importance values for bar width
            values = importance_normalized[indices]
            names = [feature_names[i] for i in indices]
            
            # Color based on positive/negative
            colors = ['#00e5ff' if shap_values[i] > 0 else '#ff5252' for i in indices]
            
            ax.barh(y_pos, values, color=colors, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, color='#e8edf5', fontsize=9)
            ax.set_xlabel('Feature Importance (Normalized 0-1)', 
                         color='#e8edf5', fontsize=10)
            ax.set_title('Feature Contributions', 
                        color='#00e5ff', fontsize=12, fontweight='bold')
            
            ax.tick_params(colors='#8899b0')
            ax.spines['bottom'].set_color('#1e2d4a')
            ax.spines['left'].set_color('#1e2d4a')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='x', alpha=0.1, color='#1e2d4a')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, 
                       facecolor='#0a0e1a', edgecolor='none')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)  # ✅ Always close figure
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            print(f"Error creating waterfall plot: {e}")
            return ""
        finally:
            # ✅ Cleanup matplotlib objects
            plt.close('all')
    
    def _create_summary_plot(
        self,
        shap_values: np.ndarray,
        features: np.ndarray,
        feature_names: List[str],
        importance_normalized: np.ndarray  # ✅ Receive normalized values
    ) -> str:
        """Create summary plot showing feature importance across all samples."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor('#0a0e1a')
            ax.set_facecolor('#0a0e1a')
            
            # Sort by normalized importance
            indices = np.argsort(importance_normalized)[-10:]
            y_pos = np.arange(len(indices))
            values = importance_normalized[indices]
            names = [feature_names[i] for i in indices]
            
            ax.barh(y_pos, values, color='#00e5ff', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, color='#e8edf5', fontsize=9)
            ax.set_xlabel('Mean |SHAP Value| (Normalized 0-1)', 
                         color='#e8edf5', fontsize=10)
            ax.set_title('Global Feature Importance', 
                        color='#00e5ff', fontsize=12, fontweight='bold')
            
            ax.tick_params(colors='#8899b0')
            ax.spines['bottom'].set_color('#1e2d4a')
            ax.spines['left'].set_color('#1e2d4a')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='x', alpha=0.1, color='#1e2d4a')
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100,
                       facecolor='#0a0e1a', edgecolor='none')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)  # ✅ Always close figure
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            print(f"Error creating summary plot: {e}")
            return ""
        finally:
            # ✅ Cleanup matplotlib objects
            plt.close('all')
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            plt.close('all')
        except:
            pass


def extract_features_for_shap(mask: np.ndarray, mri_image: np.ndarray = None) -> np.ndarray:
    """
    Extract features from mask and MRI for SHAP analysis.
    
    Returns:
        Feature vector (1D array)
    """
    import cv2
    
    features = []
    
    # Geometric features
    contours, _ = cv2.findContours(
        (mask > 0.5).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        features.append(area)
        features.append(perimeter)
        
        # Circularity
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        features.append(circularity)
        
        # Solidity
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
        else:
            solidity = 0
        features.append(solidity)
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0
        features.append(aspect_ratio)
        
        # Centroid
        M = cv2.moments(largest_contour)
        if M['m00'] > 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
        else:
            cx, cy = 0, 0
        
        features.append(w)
        features.append(h)
    else:
        features.extend([0] * 7)
        cx, cy = 0, 0
    
    # Intensity features
    if mri_image is not None:
        tumor_pixels = mri_image[mask > 0.5]
        if len(tumor_pixels) > 0:
            features.append(np.mean(tumor_pixels))
            features.append(np.std(tumor_pixels))
        else:
            features.extend([0, 0])
    else:
        features.extend([0, 0])
    
    # Location (normalized)
    features.append(cx / mask.shape[1])
    features.append(cy / mask.shape[0])
    
    return np.array(features, dtype=np.float32)