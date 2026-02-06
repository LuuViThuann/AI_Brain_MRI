"""
gradcam.py (FIXED VERSION)
Grad-CAM visualization for CNN explanations.

FIXES:
1. Matplotlib backend set to Agg for FastAPI threading
2. Tensor assignment error - convert to numpy before operations
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
import tensorflow as tf
from PIL import Image
import cv2


class GradCAMExplainer:
    """
    Generate Grad-CAM heatmaps to visualize CNN attention.
    
    FIXES:
    - Thread-safe matplotlib backend for FastAPI async
    - Proper tensor handling to avoid assignment errors
    """
    
    def __init__(self, model, layer_name=None):
        """
        Args:
            model: Trained Keras model
            layer_name: Name of convolutional layer to visualize
                       (default: last conv layer)
        """
        self.model = model
        
        # Auto-detect last conv layer if not specified
        if layer_name is None:
            layer_name = self._find_last_conv_layer()
        
        self.layer_name = layer_name
        self.grad_model = self._build_grad_model()
        
        self.layer_info = {
            "layer_name": self.layer_name,
            "layer_type": "Conv2D",
            "position": "Encoder bottleneck (last downsampling layer)",
            "feature_dimension": None  # Will be set after first run
        }

        print(f"[GradCAM] Initialized with matplotlib backend: {matplotlib.get_backend()}")
    
    def _find_last_conv_layer(self):
        """Find the last convolutional layer in the model."""
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        raise ValueError("No convolutional layer found in model")
    
    def _build_grad_model(self):
        """Build gradient model for Grad-CAM."""
        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
        return grad_model
    
    def generate_gradcam(self, img_array):
        """
        Generate Grad-CAM heatmap for input image.
        
        Args:
            img_array: np.array of shape (1, 256, 256, 1) OR (256, 256, 1)
        
        Returns:
            dict with:
                - heatmap: np.array (256, 256) normalized 0-1
                - overlay: PIL.Image - original + heatmap
                - attention_score: float - overall attention strength
                - focused_regions: list of bounding boxes
        """
        # Ensure correct shape (1, 256, 256, 1)
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        # Convert to tensor if needed
        if not isinstance(img_array, tf.Tensor):
            img_array = tf.constant(img_array, dtype=tf.float32)
        
        if self.layer_info["feature_dimension"] is None:
            with tf.GradientTape() as tape:
                conv_outputs, predictions = self.grad_model(img_array)
            self.layer_info["feature_dimension"] = conv_outputs.shape[-1]

         # ===== ENHANCED METADATA =====
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            # Use mean of prediction as the score
            loss = tf.reduce_mean(predictions)
        
        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # ✅ FIX: Convert to numpy BEFORE any operations
        conv_outputs_np = conv_outputs[0].numpy()
        pooled_grads_np = pooled_grads.numpy()
        
        # Weight feature maps by gradients (now safe - working with numpy)
        weighted_features = np.zeros_like(conv_outputs_np)
        for i in range(pooled_grads_np.shape[-1]):
            weighted_features[:, :, i] = conv_outputs_np[:, :, i] * pooled_grads_np[i]
        
        # Average across channels to get heatmap
        heatmap = np.mean(weighted_features, axis=-1)
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)  # ReLU
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Resize to original image size
        heatmap_resized = cv2.resize(heatmap, (256, 256))
        
          # Calculate attention statistics
       


        # Get original image for overlay
        if isinstance(img_array, tf.Tensor):
            original_img = img_array.numpy()[0, :, :, 0]
        else:
            original_img = img_array[0, :, :, 0]
        
        # Generate overlay
        overlay = self._create_overlay(original_img, heatmap_resized)
        
        # Compute attention score
        attention_score = float(np.mean(heatmap_resized[heatmap_resized > 0.5]))
        
        # Find focused regions
        focused_regions = self._find_focused_regions(heatmap_resized)

        interpretation = self._interpret_attention_score(attention_score)
        
        attention_stats = {
            "mean_attention": float(np.mean(heatmap_resized)),
            "max_attention": float(np.max(heatmap_resized)),
            "attention_coverage": float(np.sum(heatmap_resized > 0.5) / heatmap_resized.size),
            "focused_regions_count": len(focused_regions)
        }

        return {
            "heatmap": heatmap_resized,
            "overlay": overlay,
            "attention_score": attention_score,
            "focused_regions": focused_regions,
            
            # ===== NEW FIELDS =====
            "technical_info": {
                **self.layer_info,
                "aggregation_method": "2D spatial (single slice)",
                "gradient_method": "Grad-CAM (Gradient-weighted Class Activation Mapping)",
                "normalization": "ReLU + max normalization"
            },
            "attention_statistics": attention_stats,
            "interpretation": interpretation,
            "confidence_level": self._get_confidence_level(attention_score),
            "slice_info": {
                "type": "axial",
                "layer_applied": self.layer_name,
                "resolution": "256x256"
            }
        }
    
    def _interpret_attention_score(self, score: float) -> str:
        """Provide clinical interpretation of attention score."""
        if score > 0.8:
            return "High confidence: CNN shows strong, focused attention on identified region"
        elif score > 0.6:
            return "Moderate-high confidence: CNN attention is concentrated but with some diffusion"
        elif score > 0.4:
            return "Moderate confidence: CNN shows distributed attention across multiple regions"
        elif score > 0.2:
            return "Low-moderate confidence: CNN attention is diffuse, prediction may be uncertain"
        else:
            return "Low confidence: CNN attention is highly diffuse, high uncertainty in prediction"

    def _get_confidence_level(self, score: float) -> str:
        """Get categorical confidence level."""
        if score > 0.7:
            return "HIGH"
        elif score > 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def _create_overlay(self, original_img, heatmap, alpha=0.4):
        """
        Create overlay of heatmap on original image.
        
        Args:
            original_img: (256, 256) grayscale
            heatmap: (256, 256) normalized 0-1
            alpha: transparency of heatmap
        
        Returns:
            PIL.Image (RGB)
        """
        # Convert grayscale to RGB
        img_rgb = np.stack([original_img] * 3, axis=-1)
        img_rgb = (img_rgb * 255).astype(np.uint8)
        
        # Apply colormap to heatmap (jet colormap)
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend
        overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored, alpha, 0)
        
        return Image.fromarray(overlay)
    
    def _find_focused_regions(self, heatmap, threshold=0.6):
        """
        Find bounding boxes of highly attended regions.
        
        Args:
            heatmap: (256, 256) normalized 0-1
            threshold: attention threshold
        
        Returns:
            List of bounding boxes
        """
        # Threshold heatmap
        binary = (heatmap > threshold).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Convert contours to bounding boxes
        regions = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small regions
                x, y, w, h = cv2.boundingRect(contour)
                regions.append({
                    "bbox": (x, y, x + w, y + h),
                    "area": w * h,
                    "attention": float(np.mean(heatmap[y:y+h, x:x+w]))
                })
        
        # Sort by attention score
        regions.sort(key=lambda r: r['attention'], reverse=True)
        
        return regions[:5]  # Return top 5 regions
    
    def compare_with_mask(self, heatmap, ground_truth_mask):
        """
        Compare Grad-CAM heatmap with ground truth tumor mask.
        
        Args:
            heatmap: (256, 256) Grad-CAM output
            ground_truth_mask: (256, 256) binary mask
        
        Returns:
            dict with overlap metrics
        """
        # Threshold heatmap to binary
        heatmap_binary = (heatmap > 0.5).astype(np.float32)
        mask_binary = (ground_truth_mask > 0.5).astype(np.float32)
        
        # Compute overlap
        intersection = np.sum(heatmap_binary * mask_binary)
        union = np.sum(np.maximum(heatmap_binary, mask_binary))
        
        iou = intersection / union if union > 0 else 0
        
        # Compute precision/recall
        tp = intersection
        fp = np.sum(heatmap_binary) - intersection
        fn = np.sum(mask_binary) - intersection
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return {
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "overlap_percentage": float(intersection / np.sum(mask_binary) * 100) if np.sum(mask_binary) > 0 else 0
        }
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            plt.close('all')
        except:
            pass


# ===== STANDALONE TEST =====
if __name__ == "__main__":
    print("=" * 70)
    print("  Grad-CAM Explainer - Test (FIXED)")
    print("=" * 70)
    print(f"  Matplotlib backend: {matplotlib.get_backend()}")
    print(f"  Thread-safe: {matplotlib.get_backend().lower() == 'agg'}")
    print("=" * 70)
    print()
    
    # Mock model for testing
    from tensorflow.keras import layers, Model
    
    inputs = layers.Input(shape=(256, 256, 1))
    x = layers.Conv2D(32, 3, activation='relu', padding='same', name='conv1')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same', name='conv2')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv3')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    outputs = layers.Reshape((1, 1, 1))(x)
    
    # Make output match U-Net shape
    outputs = layers.UpSampling2D(size=(256, 256))(outputs)
    
    model = Model(inputs, outputs)
    
    print("Test model created")
    print(f"Last conv layer: conv3")
    print()
    
    # Create explainer
    explainer = GradCAMExplainer(model, layer_name='conv3')
    print("✅ GradCAMExplainer initialized")
    print()
    
    # Generate fake input
    fake_img = np.random.rand(1, 256, 256, 1).astype(np.float32)
    print("Generating Grad-CAM...")
    
    result = explainer.generate_gradcam(fake_img)
    
    print("✅ Grad-CAM generated successfully")
    print(f"   Heatmap shape: {result['heatmap'].shape}")
    print(f"   Attention score: {result['attention_score']:.3f}")
    print(f"   Focused regions: {len(result['focused_regions'])}")
    print()
    
    print("✅ Test complete - NO ERRORS!")