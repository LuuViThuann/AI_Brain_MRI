"""
visualization.py
Utility functions for visualizing MRI predictions:
  - Segmentation mask overlay (heatmap on original image)
  - Side-by-side comparison (original | mask | overlay)
  - Confidence heatmap from raw prediction scores
  - Save/export plots as PNG

Usage:
    from utils.visualization import (
        overlay_mask,
        generate_heatmap,
        side_by_side,
        save_comparison
    )
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Optional: matplotlib for richer plots (used in save_comparison)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend (no display needed)
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================
# COLOR CONSTANTS
# ============================================================
TUMOR_COLOR_RGBA   = (255, 82, 82, 140)    # Red, semi-transparent
EDGE_COLOR_RGBA    = (255, 145, 0, 100)    # Orange, semi-transparent
HEALTHY_COLOR_RGB  = (26, 107, 138)        # Teal
TUMOR_COLOR_RGB    = (255, 82, 82)         # Red
OVERLAY_ALPHA      = 0.45                  # Default overlay transparency


# ============================================================
# 1. OVERLAY MASK ON ORIGINAL IMAGE
# ============================================================
def overlay_mask(
    original: Image.Image,
    mask: np.ndarray,
    alpha: float = OVERLAY_ALPHA,
    tumor_color: tuple = (255, 82, 82),
    edge_color: tuple  = (255, 145, 0),
    edge_width: int    = 3
) -> Image.Image:
    """
    Overlay a binary segmentation mask onto the original MRI image.

    Args:
        original:    PIL Image (grayscale or RGB), any size
        mask:        numpy array (H, W) with values 0 or 1
        alpha:       transparency of the overlay (0.0 = invisible, 1.0 = opaque)
        tumor_color: RGB tuple for tumor region fill
        edge_color:  RGB tuple for tumor boundary
        edge_width:  pixel width of the boundary outline

    Returns:
        PIL Image (RGBA) with colored overlay
    """
    # Ensure original is RGB
    if original.mode != 'RGB':
        original = original.convert('RGB')

    # Resize mask to match original if needed
    h, w = original.size[1], original.size[0]
    if mask.shape != (h, w):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_img = mask_img.resize((w, h), Image.NEAREST)
        mask = np.array(mask_img) / 255.0

    # Create overlay layer (RGBA)
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    overlay_arr = np.array(overlay)

    # Fill tumor region
    tumor_pixels = mask > 0.5
    overlay_arr[tumor_pixels] = [
        tumor_color[0],
        tumor_color[1],
        tumor_color[2],
        int(alpha * 255)
    ]

    # Detect and color edges (boundary pixels)
    edges = detect_edges(mask)
    overlay_arr[edges] = [
        edge_color[0],
        edge_color[1],
        edge_color[2],
        int(0.8 * 255)  # Edges are more opaque
    ]

    # Composite: original + overlay
    original_rgba = original.convert('RGBA')
    overlay_img   = Image.fromarray(overlay_arr, mode='RGBA')
    result        = Image.alpha_composite(original_rgba, overlay_img)

    return result.convert('RGB')


def detect_edges(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """
    Detect boundary pixels of a binary mask using neighbor comparison.
    A pixel is an edge if it's inside the mask but has at least one
    neighbor outside the mask.

    Args:
        mask:      (H, W) binary array (0 or 1)
        thickness: how many pixels inward to mark as edge

    Returns:
        (H, W) boolean array — True for edge pixels
    """
    h, w = mask.shape
    edges = np.zeros_like(mask, dtype=bool)
    binary = mask > 0.5

    for t in range(1, thickness + 1):
        # Shift in 4 directions and check neighbors
        up    = np.zeros_like(binary); up[t:, :]   = binary[:-t, :]
        down  = np.zeros_like(binary); down[:-t, :] = binary[t:, :]
        left  = np.zeros_like(binary); left[:, t:]  = binary[:, :-t]
        right = np.zeros_like(binary); right[:, :-t]= binary[:, t:]

        # Edge = inside mask AND at least one neighbor is outside
        neighbor_outside = (~up) | (~down) | (~left) | (~right)
        edges |= (binary & neighbor_outside)

    return edges


# ============================================================
# 2. GENERATE CONFIDENCE HEATMAP
# ============================================================
def generate_heatmap(
    prediction_scores: np.ndarray,
    colormap: str = 'hot'
) -> Image.Image:
    """
    Convert raw CNN prediction scores (0–1 float map) into a
    color heatmap image.

    Args:
        prediction_scores: (H, W) float array, values in [0, 1]
        colormap:          'hot', 'cool', 'plasma', 'viridis'

    Returns:
        PIL Image (RGB) — color-coded heatmap
    """
    scores = np.clip(prediction_scores, 0, 1).astype(np.float32)
    h, w = scores.shape

    if HAS_MATPLOTLIB:
        # Use matplotlib colormaps for richer output
        cmap = plt.get_cmap(colormap)
        heatmap_rgb = (cmap(scores)[:, :, :3] * 255).astype(np.uint8)
        return Image.fromarray(heatmap_rgb, mode='RGB')

    else:
        # Fallback: manual red-black gradient
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = (scores * 255).astype(np.uint8)  # Red channel = intensity
        rgb[:, :, 1] = (scores * 80).astype(np.uint8)   # Slight green tint
        return Image.fromarray(rgb, mode='RGB')


# ============================================================
# 3. SIDE-BY-SIDE COMPARISON
# ============================================================
def side_by_side(
    original: Image.Image,
    mask: np.ndarray,
    prediction_scores: np.ndarray = None,
    size: int = 256
) -> Image.Image:
    """
    Create a side-by-side comparison image:
        [Original] [Mask] [Overlay] [Heatmap (optional)]

    Args:
        original:           PIL Image (grayscale/RGB)
        mask:               (H, W) binary array
        prediction_scores:  (H, W) float array (optional, for heatmap)
        size:               output size per panel

    Returns:
        PIL Image (RGB) — horizontal strip of panels
    """
    panels = []

    # Panel 1: Original
    orig_rgb = original.convert('RGB').resize((size, size), Image.LANCZOS)
    panels.append(orig_rgb)

    # Panel 2: Mask (white on black)
    mask_vis = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
    mask_vis = mask_vis.resize((size, size), Image.NEAREST).convert('RGB')
    panels.append(mask_vis)

    # Panel 3: Overlay
    overlay_img = overlay_mask(original, mask)
    overlay_img = overlay_img.resize((size, size), Image.LANCZOS)
    panels.append(overlay_img)

    # Panel 4: Heatmap (if scores provided)
    if prediction_scores is not None:
        heatmap = generate_heatmap(prediction_scores)
        heatmap = heatmap.resize((size, size), Image.LANCZOS)
        panels.append(heatmap)

    # Stitch horizontally
    total_w = size * len(panels)
    result  = Image.new('RGB', (total_w, size), (10, 14, 26))

    for i, panel in enumerate(panels):
        result.paste(panel, (i * size, 0))

    # Add labels
    result = add_panel_labels(result, panels, size)

    return result


def add_panel_labels(
    img: Image.Image,
    panels: list,
    panel_width: int
) -> Image.Image:
    """Draw text labels below each panel."""
    labels = ['Original', 'Mask', 'Overlay', 'Heatmap']
    label_height = 28
    w, h = img.size

    # Extend image height for labels
    labeled = Image.new('RGB', (w, h + label_height), (10, 14, 26))
    labeled.paste(img, (0, 0))

    draw = ImageDraw.Draw(labeled)

    # Try to load a font; fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for i in range(len(panels)):
        label = labels[i] if i < len(labels) else ''
        x = i * panel_width + panel_width // 2
        draw.text(
            (x, h + 6),
            label,
            fill=(0, 229, 255),
            font=font,
            anchor='mt'  # middle-top anchor
        )

    return labeled


# ============================================================
# 4. SAVE COMPARISON PLOT (matplotlib version)
# ============================================================
def save_comparison(
    original: Image.Image,
    mask: np.ndarray,
    prediction_scores: np.ndarray,
    output_path: str = "comparison.png",
    title: str = "Brain MRI — Prediction Comparison"
):
    """
    Save a high-quality matplotlib figure with labeled subplots.
    Falls back to side_by_side() if matplotlib is unavailable.

    Args:
        original:           PIL Image
        mask:               (H, W) binary array
        prediction_scores:  (H, W) float array
        output_path:        where to save the PNG
        title:              figure title
    """
    if not HAS_MATPLOTLIB:
        # Fallback: use PIL side-by-side
        comparison = side_by_side(original, mask, prediction_scores)
        comparison.save(output_path)
        print(f"[✓] Saved comparison (PIL fallback) → {output_path}")
        return

    orig_arr = np.array(original.convert('L'))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    fig.patch.set_facecolor('#0a0e1a')
    fig.suptitle(title, color='#00e5ff', fontsize=14, fontweight='bold', y=1.02)

    # --- Panel 1: Original ---
    axes[0].imshow(orig_arr, cmap='gray')
    axes[0].set_title('Original MRI', color='#8899b0', fontsize=11, pad=8)
    axes[0].axis('off')

    # --- Panel 2: Predicted Mask ---
    axes[1].imshow(mask, cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title('Predicted Mask', color='#8899b0', fontsize=11, pad=8)
    axes[1].axis('off')

    # --- Panel 3: Overlay ---
    overlay = overlay_mask(original, mask)
    axes[2].imshow(np.array(overlay))
    axes[2].set_title('Overlay', color='#8899b0', fontsize=11, pad=8)
    axes[2].axis('off')

    # --- Panel 4: Confidence Heatmap ---
    im = axes[3].imshow(prediction_scores, cmap='plasma', vmin=0, vmax=1)
    axes[3].set_title('Confidence Map', color='#8899b0', fontsize=11, pad=8)
    axes[3].axis('off')
    # Small colorbar
    cbar = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='#8899b0')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8899b0', fontsize=8)

    # Style all axes backgrounds
    for ax in axes:
        ax.set_facecolor('#0a0e1a')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
    plt.close(fig)
    print(f"[✓] Saved comparison → {output_path}")


# ============================================================
# 5. QUICK DEMO (run standalone)
# ============================================================
if __name__ == "__main__":
    # Generate synthetic demo data
    print("[Demo] Generating synthetic MRI + mask...")

    size = 256

    # Fake grayscale MRI (noise + bright center)
    np.random.seed(42)
    mri_arr = np.random.normal(40, 15, (size, size)).clip(0, 255).astype(np.uint8)
    # Add a bright brain-like ellipse
    for y in range(size):
        for x in range(size):
            dist = ((x - 128) / 90)**2 + ((y - 130) / 110)**2
            if dist < 1.0:
                mri_arr[y, x] = min(255, int(mri_arr[y, x] + 80 * (1 - dist)))

    original = Image.fromarray(mri_arr, mode='L')

    # Fake tumor mask (blob in upper-left quadrant)
    mask = np.zeros((size, size), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            if (x - 100)**2 + (y - 90)**2 < 40**2:
                mask[y, x] = 1.0

    # Fake prediction scores (smooth version of mask)
    from scipy.ndimage import gaussian_filter
    prediction_scores = gaussian_filter(mask, sigma=8)
    prediction_scores = prediction_scores / prediction_scores.max()

    # --- Test each function ---
    print("[1/4] overlay_mask()...")
    overlay_img = overlay_mask(original, mask)
    overlay_img.save("demo_overlay.png")
    print("      → Saved demo_overlay.png")

    print("[2/4] generate_heatmap()...")
    heatmap = generate_heatmap(prediction_scores)
    heatmap.save("demo_heatmap.png")
    print("      → Saved demo_heatmap.png")

    print("[3/4] side_by_side()...")
    comparison = side_by_side(original, mask, prediction_scores)
    comparison.save("demo_sidebyside.png")
    print("      → Saved demo_sidebyside.png")

    print("[4/4] save_comparison()...")
    save_comparison(original, mask, prediction_scores, output_path="demo_comparison.png")

    print("\n[✓] All demo outputs saved. Open the PNG files to inspect.")