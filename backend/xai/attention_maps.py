"""
attention_maps.py (FIXED VERSION)
Visualize attention mechanisms in CNN layers.

FIX: Matplotlib backend set to Agg for FastAPI threading compatibility

Provides:
  - Multi-layer attention visualization
  - Channel-wise attention
  - Spatial attention maps
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
import io
import base64
from typing import Dict, List


class AttentionVisualizer:
    """
    Visualize attention patterns in CNN layers.
    Shows what the network "sees" at different depths.
    
    FIXED: Thread-safe matplotlib backend for FastAPI async
    """
    
    def __init__(self, model):
        """
        Args:
            model: Trained Keras model
        """
        self.model = model
        self.conv_layers = self._find_conv_layers()
        print(f"[AttentionVisualizer] Initialized with matplotlib backend: {matplotlib.get_backend()}")
    
    def _find_conv_layers(self) -> List[str]:
        """Find all convolutional layers in model."""
        conv_layers = []
        
        for layer in self.model.layers:
            if 'conv' in layer.name.lower():
                conv_layers.append(layer.name)
        
        return conv_layers
    
    def generate_attention_map(
        self,
        image: np.ndarray,
        layer_name: str = None
    ) -> Dict:
        """
        Generate attention map for a specific layer.
        
        Args:
            image: Input image (1, 256, 256, 1)
            layer_name: Layer to visualize (default: last conv layer)
        
        Returns:
            {
                "attention_map": np.array,
                "channel_attention": np.array,
                "spatial_attention": np.array,
                "visualization": base64_image
            }
        """
        if layer_name is None:
            layer_name = self.conv_layers[-1] if self.conv_layers else None
        
        if layer_name is None:
            raise ValueError("No convolutional layers found")
        
        # Get layer output
        intermediate_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output
        )
        
        activations = intermediate_model.predict(image, verbose=0)
        
        # Compute attention maps
        channel_attention = self._compute_channel_attention(activations)
        spatial_attention = self._compute_spatial_attention(activations)
        combined_attention = self._compute_combined_attention(activations)
        
        # Visualize
        viz = self._visualize_attention(
            image[0, :, :, 0],
            combined_attention,
            channel_attention,
            spatial_attention
        )
        
        return {
            "layer_name": layer_name,
            "attention_map": combined_attention.tolist(),
            "channel_attention": channel_attention.tolist(),
            "spatial_attention": spatial_attention.tolist(),
            "visualization": viz,
            "activation_shape": list(activations.shape)
        }
    
    def generate_multi_head_attention(
        self,
        image: np.ndarray,
        n_heads: int = 4
    ) -> Dict:
        """
        Generate attention from multiple layers (multi-head).
        
        Args:
            image: Input image
            n_heads: Number of layers to use as "heads"
        
        Returns:
            {
                "attention_maps": list of arrays,
                "layer_names": list of str,
                "combined_visualization": base64_image
            }
        """
        # Select layers evenly distributed
        if len(self.conv_layers) < n_heads:
            selected_layers = self.conv_layers
        else:
            step = len(self.conv_layers) // n_heads
            selected_layers = [
                self.conv_layers[i * step]
                for i in range(n_heads)
            ]
        
        attention_maps = []
        layer_names = []
        
        for layer_name in selected_layers:
            result = self.generate_attention_map(image, layer_name)
            attention_maps.append(result['attention_map'])
            layer_names.append(layer_name)
        
        # Create combined visualization
        combined_viz = self._visualize_multi_head(
            image[0, :, :, 0],
            attention_maps,
            layer_names
        )
        
        return {
            "attention_maps": attention_maps,
            "layer_names": layer_names,
            "combined_visualization": combined_viz,
            "n_heads": len(selected_layers)
        }
    
    def _compute_channel_attention(self, activations: np.ndarray) -> np.ndarray:
        """
        Compute channel-wise attention (which channels are important).
        
        Returns:
            1D array of channel importances
        """
        # Global average pooling across spatial dimensions
        channel_importance = np.mean(activations, axis=(0, 1, 2))
        
        # Normalize
        channel_importance = channel_importance / (np.max(channel_importance) + 1e-8)
        
        return channel_importance
    
    def _compute_spatial_attention(self, activations: np.ndarray) -> np.ndarray:
        """
        Compute spatial attention (which locations are important).
        
        Returns:
            2D array (H, W) of spatial importances
        """
        # Average across channels
        spatial_importance = np.mean(activations, axis=-1)[0]
        
        # Normalize
        spatial_importance = spatial_importance / (np.max(spatial_importance) + 1e-8)
        
        return spatial_importance
    
    def _compute_combined_attention(self, activations: np.ndarray) -> np.ndarray:
        """
        Compute combined attention map.
        
        Returns:
            2D array (H, W)
        """
        # Weight channels by their importance
        channel_weights = self._compute_channel_attention(activations)
        
        # Weighted sum across channels
        weighted_activations = np.zeros(activations.shape[1:3])
        
        for i in range(activations.shape[-1]):
            weighted_activations += activations[0, :, :, i] * channel_weights[i]
        
        # Normalize
        weighted_activations = np.maximum(weighted_activations, 0)
        if np.max(weighted_activations) > 0:
            weighted_activations = weighted_activations / np.max(weighted_activations)
        
        # Resize to input size
        attention_resized = cv2.resize(weighted_activations, (256, 256))
        
        return attention_resized
    
    def _visualize_attention(
        self,
        original: np.ndarray,
        attention: np.ndarray,
        channel_attn: np.ndarray,
        spatial_attn: np.ndarray
    ) -> str:
        """Create visualization of attention maps."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            fig.patch.set_facecolor('#0a0e1a')
            
            # Original image
            axes[0, 0].imshow(original, cmap='gray')
            axes[0, 0].set_title('Original MRI', color='#00e5ff', fontsize=10)
            axes[0, 0].axis('off')
            
            # Attention overlay
            overlay = self._create_attention_overlay(original, attention)
            axes[0, 1].imshow(overlay)
            axes[0, 1].set_title('Attention Overlay', color='#00e5ff', fontsize=10)
            axes[0, 1].axis('off')
            
            # Channel attention (bar chart)
            axes[1, 0].barh(range(len(channel_attn[:20])), channel_attn[:20], color='#00e5ff')
            axes[1, 0].set_xlabel('Importance', color='#e8edf5', fontsize=8)
            axes[1, 0].set_ylabel('Channel', color='#e8edf5', fontsize=8)
            axes[1, 0].set_title('Channel Attention (top 20)', color='#00e5ff', fontsize=10)
            axes[1, 0].set_facecolor('#0a0e1a')
            axes[1, 0].tick_params(colors='#8899b0', labelsize=6)
            
            # Spatial attention heatmap
            spatial_resized = cv2.resize(spatial_attn, (256, 256))
            axes[1, 1].imshow(spatial_resized, cmap='hot')
            axes[1, 1].set_title('Spatial Attention', color='#00e5ff', fontsize=10)
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, 
                       facecolor='#0a0e1a', bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)  # ✅ Always close figure
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            print(f"Error visualizing attention: {e}")
            return ""
        finally:
            # ✅ Cleanup matplotlib objects
            plt.close('all')
    
    def _create_attention_overlay(
        self,
        original: np.ndarray,
        attention: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Create attention overlay on original image."""
        # Convert to RGB
        img_rgb = np.stack([original] * 3, axis=-1)
        img_rgb = (img_rgb * 255).astype(np.uint8)
        
        # Apply colormap to attention
        attention_colored = cv2.applyColorMap(
            (attention * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        attention_colored = cv2.cvtColor(attention_colored, cv2.COLOR_BGR2RGB)
        
        # Blend
        overlay = cv2.addWeighted(img_rgb, 1 - alpha, attention_colored, alpha, 0)
        
        return overlay
    
    def _visualize_multi_head(
        self,
        original: np.ndarray,
        attention_maps: List[np.ndarray],
        layer_names: List[str]
    ) -> str:
        """Visualize multi-head attention."""
        try:
            n_heads = len(attention_maps)
            fig, axes = plt.subplots(1, n_heads + 1, figsize=(4 * (n_heads + 1), 4))
            fig.patch.set_facecolor('#0a0e1a')
            
            # Original
            axes[0].imshow(original, cmap='gray')
            axes[0].set_title('Original', color='#00e5ff', fontsize=9)
            axes[0].axis('off')
            
            # Each head
            for i, (attn_map, layer_name) in enumerate(zip(attention_maps, layer_names)):
                attn_array = np.array(attn_map)
                axes[i + 1].imshow(attn_array, cmap='hot')
                axes[i + 1].set_title(f'{layer_name}\n(Head {i+1})', 
                                     color='#00e5ff', fontsize=8)
                axes[i + 1].axis('off')
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100,
                       facecolor='#0a0e1a', bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)  # ✅ Always close figure
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            print(f"Error visualizing multi-head: {e}")
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


# ===== STANDALONE TEST =====
if __name__ == "__main__":
    print("=" * 70)
    print("  Attention Visualizer - Test (FIXED)")
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
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    print("Test model created")
    print(f"Conv layers: {[l.name for l in model.layers if 'conv' in l.name]}")
    print()
    
    # Create visualizer
    visualizer = AttentionVisualizer(model)
    print(f"✅ Visualizer initialized with {len(visualizer.conv_layers)} conv layers")
    print()
    
    # Generate fake input
    fake_img = np.random.rand(1, 256, 256, 1).astype(np.float32)
    
    # Generate attention
    print("Generating attention map...")
    result = visualizer.generate_attention_map(fake_img, layer_name='conv3')
    
    print("✅ Attention map generated")
    print(f"   Layer: {result['layer_name']}")
    print(f"   Activation shape: {result['activation_shape']}")
    print(f"   Channel attention shape: {len(result['channel_attention'])}")
    print(f"   Spatial attention shape: {np.array(result['spatial_attention']).shape}")
    print()
    
    # Multi-head attention
    print("Generating multi-head attention...")
    multi_result = visualizer.generate_multi_head_attention(fake_img, n_heads=3)
    
    print("✅ Multi-head attention generated")
    print(f"   Heads: {multi_result['n_heads']}")
    print(f"   Layers: {multi_result['layer_names']}")
    print()
    
    print("✅ Test complete - NO TKINTER ERRORS!")