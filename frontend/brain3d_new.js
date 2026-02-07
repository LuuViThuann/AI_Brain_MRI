/**
 * brain3d.js (ENHANCED VERSION V4 - PROMINENT TUMOR VISUALIZATION)
 * Three.js 3D Brain Visualization Engine
 *
 * MAJOR IMPROVEMENTS:
 * ✅ VIVID tumor colors - Bright neon red, orange glow, pulsing effects
 * ✅ LARGER tumor particles - 2x-3x bigger, highly visible
 * ✅ DUAL-LAYER glow system - Core red + outer neon halo
 * ✅ AGGRESSIVE pulsing animation - Dynamic breathing effect
 * ✅ Enhanced vertex coloring - Tumor highlighting with sharp edges
 * ✅ Cone/spike visual markers - Pointing to tumor center
 */

(function BrainViewer() {

  // ---- Configuration ----
  const BRAIN_MODEL_PATH = '/frontend/models/Brain.glb';
  const DETAIL_BRAIN_MODEL_PATH = '/frontend/models/detail_brain.glb';
  const USE_FALLBACK_IF_MISSING = true;
  const FORCE_FALLBACK = false;
  
  // Color Configuration - VIVID Medical Colors
  const BRAIN_COLORS = {
    // Realistic brain tissue pink/beige color
    healthy: 0xE8B4A8,      // Soft pink-beige (like real brain)
    healthyDark: 0xD89B8F,  // Darker pink for wrinkles
    
    // Tumor colors - HIGHLY VISIBLE NEON
    tumorCore: 0xFF0040,       // Vivid neon red
    tumorMid: 0xFF1744,        // Bright red
    tumorEdge: 0xFF6B00,       // Neon orange
    tumorGlow: 0xFF3D00,       // Deep orange glow
    tumorIntense: 0xFF00FF,    // Magenta highlight for core
    tumorHalo: 0xFF4500       // Orange-red halo
  };
  
  // ---- State ----
  let scene, camera, renderer, brainMesh, tumorParticles, tumorGlow, tumorSpikes;
  let animationId = null;
  let isAutoRotating = true;
  let isDetailView = false;
  let mouseX = 0, mouseY = 0;
  let targetRotX = 0, targetRotY = 0;
  let currentRotX = 0, currentRotY = 0;
  let isDragging = false;
  let lastMouseX = 0, lastMouseY = 0;
  let clock;
  let loader;
  let originalBrainMaterials = [];
  let loadedTextures = {};
  
  // Model caching
  let cachedBrainModel = null;
  let cachedDetailBrainModel = null;
  let currentModelType = 'normal'; // 'normal' or 'detail'
  let isLoadingModel = false;
  
  // Track scale factors for each model (for proper tumor alignment)
  let normalBrainScale = 1.0;
  let detailBrainScale = 1.0;
  
  // Tumor data for re-application
  let currentTumorPoints = null;
  let tumorContour = null;  // NEW: Red contour/boundary mesh
  
  let depthVectorGroup = null;

  let tumorTooltip = null;                    
  let isTooltipVisible = false;              
  let mouseOverTumor = false;                
  let lastTumorHoverData = null;

  // ---- Init ----
  window.initBrainViewer = function() {
    console.log('%c[Brain3D] 🧠 Initializing Enhanced Brain Viewer V4 (VIVID TUMOR)...', 'color: #ff1744; font-weight: bold;');
    
    const container = document.getElementById('viewer3d');
    const canvas    = document.getElementById('brainCanvas');

    if (!container || !canvas) {
      console.error('[Brain3D] ❌ 3D viewer container not found');
      return;
    }

    // Scene with warmer background for medical feel
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e1a);
    scene.fog = new THREE.Fog(0x0a0e1a, 12, 25);

    // Camera
    const w = container.clientWidth;
    const h = container.clientHeight || 400;
    camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 100);
    camera.position.set(0, 0.8, 5.0);
    camera.lookAt(0, 0, 0);

    // Renderer with enhanced settings for textures
    renderer = new THREE.WebGLRenderer({
      canvas: canvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.3;  // Slightly brighter for tumor glow
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.gammaFactor = 2.2;

    // Enhanced Medical Lighting with more brightness for tumor
    setupRealisticLighting();

    // GLTFLoader
    if (typeof THREE.GLTFLoader !== 'undefined') {
      loader = new THREE.GLTFLoader();
    }

    // Load normal brain model first
    loadBrainModel('normal');

    // Clock for animation
    clock = new THREE.Clock();

    // Event listeners
    setupEvents(container);

    // Start render loop
    animate();

    createTumorTooltip(); 

    // Resize handler
    window.addEventListener('resize', () => {
      const w2 = container.clientWidth;
      const h2 = container.clientHeight || 400;
      camera.aspect = w2 / h2;
      camera.updateProjectionMatrix();
      renderer.setSize(w2, h2);
    });

    console.log('[Brain3D] ✅ Enhanced brain viewer V4 initialized with VIVID tumor effects');
  };

  // ---- Hide Loading Indicator ----
  function hideModelLoading() {
    const loadingEl = document.getElementById('modelLoading');
    if (loadingEl) {
      setTimeout(() => {
        loadingEl.style.display = 'none';
      }, 1000);
    }
  }

  function createTumorTooltip() {
    console.log('[Brain3D] 🏷️  Creating tumor tooltip element...');
    
    const tooltip = document.createElement('div');
    tooltip.id = 'tumorTooltip';
    
    tooltip.style.cssText = `
      position: fixed;
      background: rgba(10, 14, 26, 0.98);
      border: 2px solid #ff0040;
      border-radius: 12px;
      padding: 14px 16px;
      font-family: 'Consolas', 'Courier New', monospace;
      font-size: 12px;
      color: #c1cfe8;
      z-index: 20;
      pointer-events: none;
      user-select: none;
      white-space: nowrap;
      box-shadow: 0 0 30px rgba(255, 0, 64, 0.5);
      display: none;
      min-width: 280px;
      animation: tooltipFadeIn 0.3s ease-out;
    `;
    
    document.body.appendChild(tooltip);
    tumorTooltip = tooltip;
    
    console.log('[Brain3D] ✅ Tooltip element created');
  }

  function showTumorTooltip(screenX, screenY, tumorData) {
    if (!tumorTooltip) {
      createTumorTooltip();
    }
    
    console.log('[Brain3D] 💬 Showing tumor tooltip:', tumorData);
    
    const metrics = tumorData || lastTumorHoverData || {};
    const depthMetrics = window.lastDepthMetrics || {};
    
    // Build HTML
    let html = `
      <div style="color: #ff0040; font-weight: bold; margin-bottom: 8px; font-size: 13px;">
        🎯 TUMOR INFORMATION
      </div>
      
      <!-- Depth Info -->
      <div style="
        padding: 8px;
        background: rgba(${depthMetrics.depth_category?.category === 'SUPERFICIAL' ? '255, 0, 64' : 
                         depthMetrics.depth_category?.category === 'SHALLOW' ? '255, 145, 0' : 
                         '255, 255, 0'}, 0.15);
        border-radius: 4px;
        margin-bottom: 8px;
        border-left: 2px solid #ff9100;
      ">
        <div style="color: #5a7a99; font-size: 10px;">DEPTH</div>
        <div style="color: #ffb74d; font-weight: bold;">
          ${depthMetrics.tumor_depth_mm?.toFixed(1) || 'N/A'} mm
        </div>
        <div style="color: #ff9100; font-size: 10px; margin-top: 2px;">
          ${depthMetrics.depth_category?.label || 'N/A'}
        </div>
      </div>
      
      <!-- Volume -->
      <div style="color: #00e5ff; margin-bottom: 6px; font-size: 11px;">
        <span style="color: #5a7a99;">📦 Volume:</span> ${metrics.volume_cm3?.toFixed(2) || 'N/A'} cm³
      </div>
      
      <!-- Area -->
      <div style="color: #00e5ff; margin-bottom: 6px; font-size: 11px;">
        <span style="color: #5a7a99;">📐 Area:</span> ${metrics.area_mm2?.toFixed(1) || 'N/A'} mm²
      </div>
      
      <!-- Centroid -->
      <div style="color: #00e5ff; margin-bottom: 6px; font-size: 11px;">
        <span style="color: #5a7a99;">🎯 Center:</span>
        <br/>
        <div style="margin-left: 6px; font-family: monospace; color: #66bb6a;">
          X: ${metrics.centroid_mm?.[0]?.toFixed(1) || 'N/A'} mm<br/>
          Y: ${metrics.centroid_mm?.[1]?.toFixed(1) || 'N/A'} mm<br/>
          Z: ${metrics.centroid_mm?.[2]?.toFixed(1) || 'N/A'} mm
        </div>
      </div>
      
      <!-- Cortex Distance -->
      <div style="color: #9c27b0; margin-bottom: 6px; font-size: 11px; border-top: 1px solid #1e3a52; padding-top: 6px;">
        <span style="color: #5a7a99;">📏 To Cortex:</span> ${metrics.distance_to_cortex_mm?.toFixed(2) || 'N/A'} mm
      </div>
      
      <!-- Risk Level -->
      <div style="color: #ff5252; margin-top: 6px; font-size: 10px; font-weight: bold; border-top: 1px solid #1e3a52; padding-top: 6px;">
        ⚠️ ${metrics.cortex_proximity || 'Medium Risk'}
      </div>
    `;
    
    tumorTooltip.innerHTML = html;
    tumorTooltip.style.left = (screenX + 15) + 'px';
    tumorTooltip.style.top = (screenY + 15) + 'px';
    tumorTooltip.style.display = 'block';
    isTooltipVisible = true;
  }

  function hideTumorTooltip() {
    if (tumorTooltip) {
      tumorTooltip.style.display = 'none';
      isTooltipVisible = false;
      mouseOverTumor = false;
      console.log('[Brain3D] 💬 Tooltip hidden');
    }
  }

  function detectTumorHover(event) {
    if (!brainMesh || !currentTumorPoints || currentTumorPoints.length === 0) return;
    
    // Get mouse position
    const container = document.getElementById('viewer3d') || document.getElementById('brainCanvas');
    if (!container) return;
    
    const rect = container.getBoundingClientRect();
    const mouse = new THREE.Vector2();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    // Raycaster
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, camera);
    
    // Check intersection with brain mesh
    const intersects = raycaster.intersectObject(brainMesh);
    
    if (intersects.length > 0) {
      // Get intersection point
      const point = intersects[0].point;
      
      // Check if point is near tumor
      let minDist = Infinity;
      for (const tp of currentTumorPoints) {
        const tp3d = new THREE.Vector3(tp[0], tp[1], tp[2]);
        const dist = point.distanceTo(tp3d);
        if (dist < minDist) minDist = dist;
      }
      
      // If near tumor (within threshold)
      if (minDist < 0.15) {
        mouseOverTumor = true;
        console.log('[Brain3D] 🎯 Detected tumor hover, distance:', minDist.toFixed(3));
        
        // Show tooltip
        showTumorTooltip(event.clientX, event.clientY, window.lastDiagnosisData?.detailed_metrics);
      } else {
        mouseOverTumor = false;
        hideTumorTooltip();
      }
    } else {
      mouseOverTumor = false;
      hideTumorTooltip();
    }
  }

  function handleTumorClick(event) {
  if (!mouseOverTumor) return;
  
  console.log('[Brain3D] 🖱️  Tumor clicked!');
  
  // Show/toggle tooltip
  if (isTooltipVisible) {
    hideTumorTooltip();
  } else {
    showTumorTooltip(event.clientX, event.clientY, window.lastDiagnosisData?.detailed_metrics);
  }
}

  // ===== TUMOR METRICS PANEL (NEW) =====
function createMetricsPanel() {
  const panelHTML = `
    <div id="tumorMetricsPanel" style="
      position: fixed;
      right: 20px;
      top: 80px;
      width: 320px;
      max-height: 600px;
      background: rgba(10, 14, 26, 0.95);
      border: 2px solid #00e5ff;
      border-radius: 12px;
      padding: 16px;
      z-index: 15;
      font-family: 'Consolas', monospace;
      font-size: 12px;
      color: #c1cfe8;
      box-shadow: 0 0 20px rgba(0, 229, 255, 0.3);
      display: none;
      overflow-y: auto;
    ">
      <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e3a52;
      ">
        <h3 style="
          color: #00e5ff;
          margin: 0;
          font-size: 14px;
          font-weight: bold;
        ">📊 Chỉ Số Khối U</h3>
        <button onclick="document.getElementById('tumorMetricsPanel').style.display='none'" 
          style="
          background: none;
          border: none;
          color: #5a7a99;
          font-size: 16px;
          cursor: pointer;
        ">✕</button>
      </div>
      
      <div id="metricsContent" style="display: flex; flex-direction: column; gap: 10px;">
        <!-- Content sẽ được inject bằng JS -->
      </div>
    </div>
  `;
  
  document.body.insertAdjacentHTML('beforeend', panelHTML);
}

// Gọi hàm này trong initBrainViewer():
createMetricsPanel();

// ===== UPDATE METRICS DISPLAY =====
window.updateTumorMetrics = function(diagnosisData) {
  const panel = document.getElementById('tumorMetricsPanel');
  if (!panel) return;
  
  const metrics = diagnosisData.detailed_metrics || {};
  const depthMetrics = diagnosisData.depth_metrics || {}; 
  const pred = diagnosisData.prediction || {};
  
  console.log('[Brain3D] 📊 Updating tumor metrics:', metrics);
  
  const html = `
   <!-- DEPTH METRICS SECTION (NEW) -->
    <div style="
      padding: 12px;
      background: ${getDepthCategoryColor(depthMetrics.depth_category?.category).bg};
      border-left: 3px solid ${getDepthCategoryColor(depthMetrics.depth_category?.category).border};
      border-radius: 4px;
      margin-bottom: 12px;
    ">
      <div style="color: #5a7a99; font-size: 10px; text-transform: uppercase; margin-bottom: 6px;">
        ${depthMetrics.depth_category?.emoji || '📏'} Độ Sâu Khối U
      </div>
      <div style="color: ${getDepthCategoryColor(depthMetrics.depth_category?.category).text}; font-size: 18px; font-weight: bold; margin-bottom: 4px;">
        ${depthMetrics.tumor_depth_mm !== undefined ? depthMetrics.tumor_depth_mm.toFixed(1) : 'N/A'} mm
      </div>
      <div style="color: ${getDepthCategoryColor(depthMetrics.depth_category?.category).text}; font-size: 11px; margin-bottom: 8px;">
        ${depthMetrics.depth_category?.label || 'N/A'}
      </div>
      
      <!-- Depth Visualization Bar -->
      <div style="width: 100%; height: 24px; background: rgba(0,0,0,0.3); border-radius: 4px; overflow: hidden; position: relative; margin-bottom: 8px;">
        ${createDepthVisualizationBar(depthMetrics.tumor_depth_mm)}
      </div>
      
      <!-- Distance from center -->
      <div style="font-size: 10px; color: #5a7a99; margin-bottom: 4px;">
        📍 Khoảng cách từ tâm não: <span style="color: #00e5ff;">${depthMetrics.distance_from_center_mm?.toFixed(1) || 'N/A'}</span> mm
      </div>
      
      <!-- Nearest cortex point -->
      <div style="font-size: 10px; color: #5a7a99;">
        🎯 Điểm gần vỏ: <span style="color: #00e5ff;">
          (${depthMetrics.nearest_cortex_point?.[0]?.toFixed(1)}, 
          ${depthMetrics.nearest_cortex_point?.[1]?.toFixed(1)}, 
          ${depthMetrics.nearest_cortex_point?.[2]?.toFixed(1)})
        </span>
      </div>
    </div>
    <!-- Thể Tích -->
    <div style="
      padding: 10px;
      background: rgba(0, 229, 255, 0.08);
      border-left: 3px solid #00e5ff;
      border-radius: 4px;
    ">
      <div style="color: #5a7a99; font-size: 10px; text-transform: uppercase; margin-bottom: 4px;">
        📦 Thể Tích
      </div>
      <div style="color: #00e5ff; font-size: 16px; font-weight: bold;">
        ${metrics.volume_cm3 !== undefined ? metrics.volume_cm3.toFixed(2) : 'N/A'}
      </div>
      <div style="color: #5a7a99; font-size: 9px;">cm³</div>
    </div>
    
    <!-- Diện Tích -->
    <div style="
      padding: 10px;
      background: rgba(255, 145, 0, 0.08);
      border-left: 3px solid #ff9100;
      border-radius: 4px;
    ">
      <div style="color: #5a7a99; font-size: 10px; text-transform: uppercase; margin-bottom: 4px;">
        📐 Diện Tích
      </div>
      <div style="color: #ff9100; font-size: 16px; font-weight: bold;">
        ${metrics.area_mm2 !== undefined ? metrics.area_mm2.toFixed(1) : 'N/A'}
      </div>
      <div style="color: #5a7a99; font-size: 9px;">mm²</div>
    </div>
    
    <!-- Tỷ Lệ U/Não -->
    <div style="
      padding: 10px;
      background: rgba(255, 82, 82, 0.08);
      border-left: 3px solid #ff5252;
      border-radius: 4px;
    ">
      <div style="color: #5a7a99; font-size: 10px; text-transform: uppercase; margin-bottom: 4px;">
        📊 Tỷ Lệ U/Não
      </div>
      <div style="color: #ff5252; font-size: 16px; font-weight: bold;">
        ${metrics.tumor_brain_ratio !== undefined ? metrics.tumor_brain_ratio.toFixed(4) : 'N/A'}
      </div>
      <div style="color: #5a7a99; font-size: 9px;">%</div>
    </div>
    
    <!-- Tọa Độ Tâm -->
    <div style="
      padding: 10px;
      background: rgba(0, 200, 83, 0.08);
      border-left: 3px solid #00c853;
      border-radius: 4px;
    ">
      <div style="color: #5a7a99; font-size: 10px; text-transform: uppercase; margin-bottom: 4px;">
        🎯 Tâm Khối U
      </div>
      <div style="color: #00c853; font-family: 'Courier New'; font-size: 11px; line-height: 1.4;">
        <div>X: ${metrics.centroid_mm ? metrics.centroid_mm[0] : 'N/A'} mm</div>
        <div>Y: ${metrics.centroid_mm ? metrics.centroid_mm[1] : 'N/A'} mm</div>
        <div>Z: ${metrics.centroid_mm ? metrics.centroid_mm[2] : 'N/A'} mm</div>
      </div>
    </div>
    
    <!-- Khoảng Cách Vỏ Não -->
    <div style="
      padding: 10px;
      background: rgba(156, 39, 176, 0.08);
      border-left: 3px solid #9c27b0;
      border-radius: 4px;
    ">
      <div style="color: #5a7a99; font-size: 10px; text-transform: uppercase; margin-bottom: 4px;">
        📏 Khoảng Cách Vỏ Não
      </div>
      <div style="color: #9c27b0; font-size: 14px; font-weight: bold;">
        ${metrics.distance_to_cortex_mm !== undefined ? metrics.distance_to_cortex_mm.toFixed(2) : 'N/A'} mm
      </div>
      <div style="color: #5a7a99; font-size: 9px;">
        ${metrics.cortex_proximity || ''}
      </div>
    </div>
    
    <!-- Divider -->
    <div style="height: 1px; background: #1e3a52; margin: 8px 0;"></div>
    
    <!-- Chỉ Số Khác -->
    <div style="color: #5a7a99; font-size: 10px;">
      <div style="margin-bottom: 6px;">
        <strong>Độ Tin Cậy:</strong> ${(pred.confidence * 100).toFixed(1)}%
      </div>
      <div style="margin-bottom: 6px;">
        <strong>Vị Trí:</strong> ${pred.location_hint || 'N/A'}
      </div>
      <div>
        <strong>Mô Hình:</strong> U-Net CNN v1.0
      </div>
    </div>
  `;
  
  document.getElementById('metricsContent').innerHTML = html;
  panel.style.display = 'block';
  
  console.log('[Brain3D] ✅ Metrics panel updated');
};

// =========== NEW UPDATE 
function createDepthVisualizationBar(depth) {
  const maxDepth = 55;  // Brain radius
  const percentage = Math.min((depth / maxDepth) * 100, 100);
  
  let color;
  if (depth < 5) color = '#ff0040';
  else if (depth < 15) color = '#ff9100';
  else if (depth < 30) color = '#ffff00';
  else if (depth < 45) color = '#00c853';
  else color = '#00a3cc';
  
  return `
    <div style="
      height: 100%;
      width: ${percentage}%;
      background: ${color};
      transition: width 0.3s ease;
      display: flex;
      align-items: center;
      justify-content: flex-end;
      padding-right: 8px;
      font-size: 10px;
      color: #0a0e1a;
      font-weight: bold;
    ">
      ${depth < 8 ? '' : (percentage < 100 ? percentage.toFixed(0) + '%' : '')}
    </div>
  `;
}

// 
function getDepthCategoryColor(category) {
  const colors = {
    'OUTSIDE': { bg: 'rgba(255, 0, 0, 0.1)', border: '#ff0000', text: '#ff5555' },
    'SUPERFICIAL': { bg: 'rgba(255, 0, 64, 0.1)', border: '#ff0040', text: '#ff5252' },
    'SHALLOW': { bg: 'rgba(255, 145, 0, 0.1)', border: '#ff9100', text: '#ffb74d' },
    'INTERMEDIATE': { bg: 'rgba(255, 255, 0, 0.1)', border: '#ffff00', text: '#ffff99' },
    'DEEP': { bg: 'rgba(0, 200, 83, 0.1)', border: '#00c853', text: '#66bb6a' },
    'VERY_DEEP': { bg: 'rgba(0, 163, 204, 0.1)', border: '#00a3cc', text: '#4dd0e1' }
  };
  
  return colors[category] || colors['INTERMEDIATE'];
}
  // ---- Show Loading Indicator ----
  function showModelLoading(message = 'Loading 3D Brain Model...') {
    const loadingEl = document.getElementById('modelLoading');
    if (loadingEl) {
      const messageEl = loadingEl.querySelector('span');
      if (messageEl) messageEl.textContent = message;
      loadingEl.style.display = 'flex';
    }
  }

  // ---- Realistic Medical Lighting Setup (Enhanced Brightness) ----
  function setupRealisticLighting() {
    // Ambient light - warmer tone for medical feel
    const ambientLight = new THREE.AmbientLight(0xffeedd, 0.8);  // Increased brightness
    scene.add(ambientLight);

    // Key light - main illumination (slightly warm)
    const keyLight = new THREE.DirectionalLight(0xfff5e6, 2.0);  // Increased intensity
    keyLight.position.set(6, 10, 8);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 4096;
    keyLight.shadow.mapSize.height = 4096;
    keyLight.shadow.camera.near = 0.5;
    keyLight.shadow.camera.far = 50;
    keyLight.shadow.bias = -0.0001;
    scene.add(keyLight);

    // Fill light - soft from side
    const fillLight = new THREE.DirectionalLight(0xffe4d4, 0.7);  // Slightly brighter
    fillLight.position.set(-5, 3, -3);
    scene.add(fillLight);

    // Rim light - for depth (cooler tone)
    const rimLight = new THREE.DirectionalLight(0xe0f0ff, 0.8);  // Brighter
    rimLight.position.set(0, -3, -6);
    scene.add(rimLight);

    // Point lights for highlights
    const pointLight1 = new THREE.PointLight(0xffd4c4, 0.7, 12);  // Brighter
    pointLight1.position.set(4, 4, 4);
    scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0xffd4c4, 0.5, 10);  // Brighter
    pointLight2.position.set(-3, -2, 3);
    scene.add(pointLight2);

    // Hemisphere light for subtle ambient occlusion feel
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.5);  // Brighter
    scene.add(hemiLight);

    // SPECIAL: Tumor-specific red accent light
    const tumorLight = new THREE.PointLight(0xFF0040, 0.6, 8);
    tumorLight.position.set(0.5, 0.6, 0.5);
    scene.add(tumorLight);
  }

  // ---- Load Brain Model (Enhanced with model type selection) ----
  function loadBrainModel(modelType = 'normal') {
    if (isLoadingModel) {
      console.warn('[Brain3D] ⚠️  Model already loading, please wait...');
      return;
    }
    
    const modelPath = modelType === 'detail' ? DETAIL_BRAIN_MODEL_PATH : BRAIN_MODEL_PATH;
    const modelName = modelType === 'detail' ? 'detail_brain.glb' : 'Brain.glb';
    
    console.log(`[Brain3D] 🔄 Loading ${modelName}...`);
    showModelLoading(`Loading ${modelName}...`);
    
    if (FORCE_FALLBACK) {
      console.log('[Brain3D] Using procedural brain (fallback mode)');
      buildProceduralBrain();
      hideModelLoading();
      return;
    }

    if (!loader) {
      console.warn('[Brain3D] GLTFLoader not available, using fallback');
      buildProceduralBrain();
      hideModelLoading();
      return;
    }
    
    // Check cache first
    if (modelType === 'normal' && cachedBrainModel) {
      console.log('[Brain3D] ✅ Using cached normal brain model');
      applyBrainModel(cachedBrainModel.clone(), modelType);
      hideModelLoading();
      return;
    }
    
    if (modelType === 'detail' && cachedDetailBrainModel) {
      console.log('[Brain3D] ✅ Using cached detail brain model');
      applyBrainModel(cachedDetailBrainModel.clone(), modelType);
      hideModelLoading();
      return;
    }
    
    isLoadingModel = true;
    
    loader.load(
      modelPath,
      (gltf) => {
        console.log(`%c[Brain3D] ✅ ${modelName} loaded successfully!`, 'color: #00ff00; font-weight: bold;');
        
        // Cache the model
        if (modelType === 'normal') {
          cachedBrainModel = gltf.scene.clone();
        } else {
          cachedDetailBrainModel = gltf.scene.clone();
        }
        
        applyBrainModel(gltf.scene, modelType);
        isLoadingModel = false;
        hideModelLoading();
      },
      (progress) => {
        if (progress.total > 0) {
          const percent = (progress.loaded / progress.total * 100).toFixed(1);
          console.log(`[Brain3D] Loading ${modelName}: ${percent}%`);
        }
      },
      (error) => {
        console.error(`[Brain3D] ❌ Error loading ${modelName}:`, error);
        isLoadingModel = false;
        
        if (USE_FALLBACK_IF_MISSING) {
          console.log('[Brain3D] Falling back to procedural brain');
          buildProceduralBrain();
        }
        hideModelLoading();
      }
    );
  }

  // ---- Apply Brain Model to Scene ----
  function applyBrainModel(model, modelType) {
    console.log(`[Brain3D] 🎨 Applying ${modelType} brain model to scene...`);
    
    // Remove old brain mesh
    if (brainMesh) {
      scene.remove(brainMesh);
      brainMesh.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (child.material.map) child.material.map.dispose();
          if (child.material.normalMap) child.material.normalMap.dispose();
          child.material.dispose();
        }
      });
    }
    
    brainMesh = model;
    currentModelType = modelType;
    
    // Scale and center
    const box = new THREE.Box3().setFromObject(brainMesh);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    
    const maxDim = Math.max(size.x, size.y, size.z);
    const scale = 2.8 / maxDim;
    
    // Track scale factor for this model (for tumor alignment)
    if (modelType === 'normal') {
      normalBrainScale = scale;
      console.log(`[Brain3D] 📐 Normal brain scale: ${scale.toFixed(3)}`);
    } else {
      detailBrainScale = scale;
      console.log(`[Brain3D] 📐 Detail brain scale: ${scale.toFixed(3)}`);
    }
    
    brainMesh.scale.multiplyScalar(scale);
    brainMesh.position.sub(center.multiplyScalar(scale));
    
    // Apply current rotation
    brainMesh.rotation.x = currentRotX;
    brainMesh.rotation.y = currentRotY;
    
    // Process materials - PRESERVE ORIGINAL TEXTURES
    originalBrainMaterials = [];
    brainMesh.traverse((child) => {
      if (child.isMesh) {
        console.log(`[Brain3D] Found mesh: ${child.name || 'Unnamed'}, Material: ${child.material.type}`);
        
        // Store original material
        originalBrainMaterials.push({
          mesh: child,
          material: child.material.clone()
        });
        
        // Enhance existing material rather than replacing
        if (child.material) {
          if (child.material.type === 'MeshStandardMaterial' || 
              child.material.type === 'MeshPhongMaterial') {
            
            child.material.metalness = 0.05;
            child.material.roughness = 0.65;
            
            if (!child.material.map && child.material.color.getHex() === 0xffffff) {
              child.material.color.setHex(BRAIN_COLORS.healthy);
            }
            
            if (!child.material.emissiveMap) {
              child.material.emissive = new THREE.Color(0x221111);
              child.material.emissiveIntensity = 0.03;
            }
            
            child.castShadow = true;
            child.receiveShadow = true;
            
            if (child.material.map) {
              child.material.map.encoding = THREE.sRGBEncoding;
              child.material.map.anisotropy = renderer.capabilities.getMaxAnisotropy();
              console.log('[Brain3D] ✅ Texture configured');
            }
            
            if (child.material.normalMap) {
              child.material.normalScale.set(1.2, 1.2);
              console.log('[Brain3D] ✅ Normal map configured');
            }
            
            child.material.needsUpdate = true;
          }
          else if (child.material.type === 'MeshBasicMaterial') {
            const oldMaterial = child.material;
            child.material = new THREE.MeshStandardMaterial({
              map: oldMaterial.map,
              color: oldMaterial.color.getHex() === 0xffffff ? BRAIN_COLORS.healthy : oldMaterial.color,
              metalness: 0.05,
              roughness: 0.65,
              emissive: new THREE.Color(0x221111),
              emissiveIntensity: 0.03
            });
            
            if (child.material.map) {
              child.material.map.encoding = THREE.sRGBEncoding;
            }
            
            child.castShadow = true;
            child.receiveShadow = true;
          }
        }
      }
    });
    
    scene.add(brainMesh);
    
    // Reapply tumor highlighting if exists
    if (currentTumorPoints && currentTumorPoints.length > 0) {
      console.log('[Brain3D] 🔄 Reapplying tumor highlighting...');
      
      // Get the scale factor for the new model
      const currentScale = (modelType === 'normal') ? normalBrainScale : detailBrainScale;
      const previousScale = (modelType === 'normal') ? detailBrainScale : normalBrainScale;
      const scaleRatio = currentScale / previousScale;
      
      console.log(`[Brain3D] 🔧 Scale ratio: ${scaleRatio.toFixed(3)} (adapting tumor to new model)`);
      
      highlightTumorRegion(currentTumorPoints);
      buildTumorParticles(currentTumorPoints);
      buildTumorSpikes(currentTumorPoints);
      buildTumorContour(currentTumorPoints);  // NEW: Build red contour
    }
    
    console.log(`[Brain3D] ✅ ${modelType} brain model applied successfully`);
  }

  // ---- Switch Brain Model ----
  function switchBrainModel(modelType) {
    if (currentModelType === modelType) {
      console.log(`[Brain3D] Already displaying ${modelType} model`);
      return;
    }
    
    console.log(`%c[Brain3D] 🔄 Switching to ${modelType} brain model...`, 'color: #00ffff; font-weight: bold;');
    loadBrainModel(modelType);
  }

  // ---- Highlight Tumor Region - ENHANCED with VIVID COLORS ----
  function highlightTumorRegion(tumorPoints) {
    if (!brainMesh) return;

    console.log(`%c[Brain3D] 🔴 HIGHLIGHTING TUMOR (${tumorPoints.length} points) - VIVID COLORS`, 'color: #FF0040; font-weight: bold; font-size: 12px;');

    const tumorColor = new THREE.Color(BRAIN_COLORS.tumorCore);
    const edgeColor = new THREE.Color(BRAIN_COLORS.tumorEdge);
    const healthyColor = new THREE.Color(BRAIN_COLORS.healthy);

    brainMesh.traverse((child) => {
      if (child.isMesh && child.geometry) {
        const positions = child.geometry.attributes.position;
        
        // Create or update vertex colors
        let colors = child.geometry.attributes.color;
        if (!colors) {
          colors = new THREE.Float32BufferAttribute(positions.count * 3, 3);
          child.geometry.setAttribute('color', colors);
        }
        
        for (let i = 0; i < positions.count; i++) {
          const vx = positions.getX(i);
          const vy = positions.getY(i);
          const vz = positions.getZ(i);

          let minDist = Infinity;
          for (const tp of tumorPoints) {
            const dx = vx - tp[0], dy = vy - tp[1], dz = vz - tp[2];
            const d = Math.sqrt(dx*dx + dy*dy + dz*dz);
            if (d < minDist) minDist = d;
          }

          // VIVID TUMOR COLORING with aggressive falloff
          if (minDist < 0.12) {
            // Core tumor - intense neon red
            colors.setXYZ(i, tumorColor.r, tumorColor.g, tumorColor.b);
          } else if (minDist < 0.25) {
            // Tumor edge gradient - sharp transition to orange
            const t = (minDist - 0.12) / 0.13;
            colors.setXYZ(i,
              edgeColor.r * (1-t) + healthyColor.r * t,
              edgeColor.g * (1-t) + healthyColor.g * t,
              edgeColor.b * (1-t) + healthyColor.b * t
            );
          } else {
            // Keep existing color or use healthy tissue color
            const existingColor = child.material.color || healthyColor;
            colors.setXYZ(i, existingColor.r, existingColor.g, existingColor.b);
          }
        }
        
        colors.needsUpdate = true;
        child.material.vertexColors = true;
        child.material.needsUpdate = true;
      }
    });
  }

  // ---- Build ENHANCED Tumor Particles ----
  function buildTumorParticles(tumorPoints) {
    console.log(`%c[Brain3D] ✨ Creating VIVID prominent tumor particles (ATTACHED to brain surface)`, 'color: #FF1744; font-weight: bold;');
    
    // Remove old
    if (tumorParticles) {
      scene.remove(tumorParticles);
      tumorParticles.geometry.dispose();
      tumorParticles.material.dispose();
    }
    if (tumorGlow) {
      scene.remove(tumorGlow);
      tumorGlow.geometry.dispose();
      tumorGlow.material.dispose();
    }

    if (!tumorPoints || tumorPoints.length === 0) return;

    // ===== MAIN TUMOR PARTICLES (Core) - SURFACE ATTACHED =====
    const particleGeo = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];
    const sizes = [];

    for (const tp of tumorPoints) {
      // FIX: Project point outward to brain surface (1.05 scale instead of 0.90-1.05)
      // This makes particles sit ON the brain surface, not inside/below it
      const dist = Math.sqrt(tp[0]*tp[0] + tp[1]*tp[1] + tp[2]*tp[2]);
      const surfaceScale = 1.05 / dist;  // Project to surface layer
      
      positions.push(tp[0] * surfaceScale, tp[1] * surfaceScale, tp[2] * surfaceScale);
      
      // VIVID COLOR VARIATION - Red to magenta
      const colorVar = Math.random();
      const r = 1.0;
      const g = 0.0 + colorVar * 0.15;  // Minimal green
      const b = 0.25 + colorVar * 0.25; // Some blue for magenta tint
      colors.push(r, g, b);
      
      // LARGER PARTICLES - 2-3x bigger than before
      sizes.push(0.06 + Math.random() * 0.08);
    }

    particleGeo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    particleGeo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    particleGeo.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));

    const particleMat = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 }
      },
      vertexShader: `
        attribute float size;
        attribute vec3 color;
        varying vec3 vColor;
        varying float vSize;
        uniform float time;
        
        void main() {
          vColor = color;
          vSize = size;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          
          // AGGRESSIVE PULSING - 1.0 to 1.6x size variation
          float pulsate = 1.0 + 0.6 * sin(time * 3.5 + position.x * 8.0);
          
          gl_PointSize = size * pulsate * 500.0 / -mvPosition.z;
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        varying float vSize;
        
        void main() {
          vec2 center = gl_PointCoord - vec2(0.5);
          float dist = length(center);
          
          if (dist > 0.5) discard;
          
          // VIVID GLOW with sharp edges
          float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
          alpha *= 1.0;  // Full opacity
          
          // Bright core with outer glow
          float brightness = 2.0 + (1.0 - dist * 2.0) * 1.5;
          
          gl_FragColor = vec4(vColor * brightness, alpha);
        }
      `,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });

    tumorParticles = new THREE.Points(particleGeo, particleMat);
    
    // Apply current rotation
    tumorParticles.rotation.x = currentRotX;
    tumorParticles.rotation.y = currentRotY;
    
    scene.add(tumorParticles);

    // ===== OUTER GLOW LAYER (Neon Halo) - SURFACE ATTACHED =====
    const glowGeo = new THREE.BufferGeometry();
    const glowPos = [];
    const glowSizes = [];
    
    for (const tp of tumorPoints) {
      // FIX: Project glow layer to surface (1.12x scale for outer halo effect)
      const dist = Math.sqrt(tp[0]*tp[0] + tp[1]*tp[1] + tp[2]*tp[2]);
      const glowSurfaceScale = 1.12 / dist;  // Slightly further out than particles
      
      glowPos.push(tp[0] * glowSurfaceScale, tp[1] * glowSurfaceScale, tp[2] * glowSurfaceScale);
      glowSizes.push(0.20 + Math.random() * 0.15);
    }
    
    glowGeo.setAttribute('position', new THREE.Float32BufferAttribute(glowPos, 3));
    glowGeo.setAttribute('size', new THREE.Float32BufferAttribute(glowSizes, 1));
    
    const glowMat = new THREE.PointsMaterial({
      color: BRAIN_COLORS.tumorGlow,
      size: 0.25,
      transparent: true,
      opacity: 0.4,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true
    });
    
    tumorGlow = new THREE.Points(glowGeo, glowMat);
    
    // Apply current rotation
    tumorGlow.rotation.x = currentRotX;
    tumorGlow.rotation.y = currentRotY;
    
    scene.add(tumorGlow);
    
    console.log('%c[Brain3D] ✅ VIVID tumor visualization COMPLETE', 'color: #FF1744; font-weight: bold;');
  }

  // ---- Build Tumor Spike Markers (NEW) ----
  function buildTumorSpikes(tumorPoints) {
    console.log(`[Brain3D] 🔱 Creating tumor spike markers...`);
    
    // Remove old spikes
    if (tumorSpikes) {
      scene.remove(tumorSpikes);
      tumorSpikes.geometry.dispose();
      tumorSpikes.material.dispose();
    }

    if (!tumorPoints || tumorPoints.length < 5) return;  // Only if enough points

    // Create cone geometry for spike markers
    const geometry = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];

    // Use subset of tumor points for spikes (every Nth point)
    const stride = Math.max(1, Math.floor(tumorPoints.length / 8));
    
    for (let i = 0; i < tumorPoints.length; i += stride) {
      const tp = tumorPoints[i];
      
      // Normalize direction and extend outward
      const mag = Math.sqrt(tp[0]*tp[0] + tp[1]*tp[1] + tp[2]*tp[2]);
      const nx = tp[0] / mag;
      const ny = tp[1] / mag;
      const nz = tp[2] / mag;
      
      // FIX: Spike base starts at brain surface (1.05 scale)
      const baseSurfaceScale = 1.05 / mag;
      const bx = tp[0] * baseSurfaceScale;
      const by = tp[1] * baseSurfaceScale;
      const bz = tp[2] * baseSurfaceScale;
      
      // Spike tip (extending further outward from surface)
      const tx = bx + nx * 0.40;
      const ty = by + ny * 0.40;
      const tz = bz + nz * 0.40;
      
      // Add line segment (base to tip)
      positions.push(bx, by, bz);
      colors.push(1.0, 0.1, 0.2);  // Red
      
      positions.push(tx, ty, tz);
      colors.push(1.0, 0.4, 0.0);  // Orange
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.LineBasicMaterial({
      vertexColors: true,
      linewidth: 3,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending
    });

    tumorSpikes = new THREE.LineSegments(geometry, material);
    
    // Apply current rotation
    tumorSpikes.rotation.x = currentRotX;
    tumorSpikes.rotation.y = currentRotY;
    
    scene.add(tumorSpikes);
    console.log(`[Brain3D] ✅ Spike markers added`);
  }

  // ---- Build Tumor Contour/Boundary (NEW) ----
  function buildTumorContour(tumorPoints) {
    console.log(`%c[Brain3D] 🔴 Creating RED CONTOUR boundary around tumor...`, 'color: #FF0040; font-weight: bold;');
    
    // Remove old contour
    if (tumorContour) {
      scene.remove(tumorContour);
      tumorContour.geometry.dispose();
      tumorContour.material.dispose();
    }

    if (!tumorPoints || tumorPoints.length < 10) return;

    // Create a convex hull or simplified boundary around tumor points
    // For simplicity, we'll create circles at different heights
    
    const geometry = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];

    // Group points by height (y-axis) to create boundary circles
    const yLevels = {};
    for (const tp of tumorPoints) {
      const yKey = Math.round(tp[1] * 10) / 10;  // Round to nearest 0.1
      if (!yLevels[yKey]) {
        yLevels[yKey] = [];
      }
      yLevels[yKey].push([tp[0], tp[2]]);  // Store x, z
    }

    // For each height level, create a boundary circle
    const segments = 12;
    const red = new THREE.Color(0xFF0040);
    
    for (const yKey in yLevels) {
      const y = parseFloat(yKey);
      const pointsAtLevel = yLevels[yKey];
      
      if (pointsAtLevel.length < 3) continue;
      
      // Find center and radius
      let cx = 0, cz = 0;
      for (const [x, z] of pointsAtLevel) {
        cx += x;
        cz += z;
      }
      cx /= pointsAtLevel.length;
      cz /= pointsAtLevel.length;
      
      // Find max radius
      let maxRadius = 0;
      for (const [x, z] of pointsAtLevel) {
        const dist = Math.sqrt((x-cx)*(x-cx) + (z-cz)*(z-cz));
        maxRadius = Math.max(maxRadius, dist);
      }
      
      // Add buffer to radius
      maxRadius *= 1.3;
      
      // Draw circle at this level
      for (let i = 0; i <= segments; i++) {
        const angle = (i / segments) * Math.PI * 2;
        const x = cx + Math.cos(angle) * maxRadius;
        const z = cz + Math.sin(angle) * maxRadius;
        
        // Project to surface
        const dist = Math.sqrt(x*x + y*y + z*z);
        const surfaceScale = 1.08 / dist;  // Slightly outside tumor particles
        
        positions.push(x * surfaceScale, y * surfaceScale, z * surfaceScale);
        colors.push(red.r, red.g, red.b);
      }
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.LineBasicMaterial({
      vertexColors: true,
      linewidth: 2,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending
    });

    tumorContour = new THREE.LineLoop(geometry, material);
    
    // Apply current rotation
    tumorContour.rotation.x = currentRotX;
    tumorContour.rotation.y = currentRotY;
    
    scene.add(tumorContour);
    console.log(`%c[Brain3D] ✅ RED CONTOUR added - visible boundary around tumor`, 'color: #FF0040; font-weight: bold;');
  }


  function buildTumorDepthVector(depthMetrics) {
    console.log(`%c[Brain3D] 📏 Building Tumor Depth Vector`, 'color: #ff9100; font-weight: bold;');
    
    if (!depthMetrics || !depthMetrics.centroid_3d) {
      console.warn('[Brain3D] ⚠️  No depth metrics available');
      return;
    }
    
    // Remove old depth vector
    if (depthVectorGroup) {
      scene.remove(depthVectorGroup);
      if (depthVectorGroup.children) {
        depthVectorGroup.children.forEach(child => {
          if (child.geometry) child.geometry.dispose();
          if (child.material) child.material.dispose();
        });
      }
    }
    
    depthVectorGroup = new THREE.Group();
    
    // ===== PARSE METRICS =====
    const centroid = depthMetrics.centroid_3d;
    const cortexPoint = depthMetrics.nearest_cortex_point;
    const depth = depthMetrics.tumor_depth_mm;
    const category = depthMetrics.depth_category || {};
    
    // Normalize to 3D space
    const scale = 1.0 / 55;  // Brain radius ≈ 55mm
    
    const tumorPos = new THREE.Vector3(
      centroid[0] * scale,
      centroid[1] * scale,
      centroid[2] * scale
    );
    
    const cortexPos = new THREE.Vector3(
      cortexPoint[0] * scale,
      cortexPoint[1] * scale,
      cortexPoint[2] * scale
    );
    
    console.log('[Brain3D] 📍 Tumor:', tumorPos);
    console.log('[Brain3D] 📍 Cortex:', cortexPos);
    console.log('[Brain3D] 📏 Depth:', depth, 'mm');
    
    // ===== GET COLOR BASED ON DEPTH =====
    const depthColor = getDepthVectorColor(category.category || 'INTERMEDIATE');
    
    // ===== 1. MAIN VECTOR LINE =====
    const lineGeom = new THREE.BufferGeometry().setFromPoints([
      cortexPos,
      tumorPos
    ]);
    
    const lineMat = new THREE.LineBasicMaterial({
      color: depthColor,
      linewidth: 4,
      transparent: true,
      opacity: 0.95
    });
    
    const depthLine = new THREE.Line(lineGeom, lineMat);
    depthVectorGroup.add(depthLine);
    
    // ===== 2. GLOW HALO AROUND LINE =====
    const glowGeom = new THREE.BufferGeometry().setFromPoints([
      cortexPos,
      tumorPos
    ]);
    
    const glowMat = new THREE.LineBasicMaterial({
      color: depthColor,
      linewidth: 12,
      transparent: true,
      opacity: 0.2,
      blending: THREE.AdditiveBlending
    });
    
    const glowLine = new THREE.Line(glowGeom, glowMat);
    depthVectorGroup.add(glowLine);
    
    // ===== 3. ARROW HEAD (pointing at tumor) =====
    const arrowGeom = new THREE.ConeGeometry(0.08, 0.2, 8);
    const arrowMat = new THREE.MeshStandardMaterial({
      color: depthColor,
      emissive: depthColor,
      emissiveIntensity: 0.8,
      metalness: 0.2,
      roughness: 0.5
    });
    
    const arrow = new THREE.Mesh(arrowGeom, arrowMat);
    arrow.position.copy(tumorPos);
    
    // Point arrow towards cortex
    const direction = new THREE.Vector3().subVectors(tumorPos, cortexPos).normalize();
    arrow.lookAt(cortexPos);
    
    depthVectorGroup.add(arrow);
    
    // ===== 4. SPHERE AT CORTEX POINT =====
    const cortexMarkerGeom = new THREE.SphereGeometry(0.05, 12, 12);
    const cortexMarkerMat = new THREE.MeshStandardMaterial({
      color: depthColor,
      emissive: depthColor,
      emissiveIntensity: 0.6,
      metalness: 0.3,
      roughness: 0.4
    });
    
    const cortexMarker = new THREE.Mesh(cortexMarkerGeom, cortexMarkerMat);
    cortexMarker.position.copy(cortexPos);
    depthVectorGroup.add(cortexMarker);
    
    // ===== 5. SPHERE AT TUMOR CENTER =====
    const tumorMarkerGeom = new THREE.SphereGeometry(0.06, 12, 12);
    const tumorMarkerMat = new THREE.MeshStandardMaterial({
      color: depthColor,
      emissive: depthColor,
      emissiveIntensity: 0.8,
      metalness: 0.3,
      roughness: 0.3
    });
    
    const tumorMarker = new THREE.Mesh(tumorMarkerGeom, tumorMarkerMat);
    tumorMarker.position.copy(tumorPos);
    depthVectorGroup.add(tumorMarker);
    
    // ===== 6. DISTANCE LABEL TEXT =====
    createDepthVectorLabel(tumorPos, depth, category, depthColor);
    
    // ===== ADD TO SCENE =====
    depthVectorGroup.rotation.x = currentRotX;
    depthVectorGroup.rotation.y = currentRotY;
    
    scene.add(depthVectorGroup);
    
    // ===== STORE FOR ANIMATION =====
    window.depthVectorPulsing = {
      line: depthLine,
      glow: glowLine,
      arrow: arrow,
      tumorMarker: tumorMarker,
      time: 0
    };
    
    console.log(`%c[Brain3D] ✅ Depth vector created: ${depth}mm (${category.label})`, 'color: #00c853; font-weight: bold;');
  }

  function getDepthVectorColor(category) {
    const colors = {
      'OUTSIDE': 0xff0000,
      'SUPERFICIAL': 0xff0040,
      'SHALLOW': 0xff9100,
      'INTERMEDIATE': 0xffff00,
      'DEEP': 0x00c853,
      'VERY_DEEP': 0x00a3cc
    };
    
    return colors[category] || 0xffff00;
  }
  function createDepthVectorLabel(worldPos, depth, category, colorHex) {
    console.log('[Brain3D] 🏷️  Creating depth vector label...');
    
    // Remove old label
    const oldLabel = document.getElementById('depthVectorLabel');
    if (oldLabel) oldLabel.remove();
    
    // Create label element
    const label = document.createElement('div');
    label.id = 'depthVectorLabel';
    
    const colorCode = '#' + colorHex.toString(16).padStart(6, '0');
    
    label.style.cssText = `
      position: fixed;
      background: rgba(10, 14, 26, 0.98);
      border: 2px solid ${colorCode};
      border-radius: 12px;
      padding: 14px 18px;
      font-family: 'Consolas', 'Courier New', monospace;
      font-size: 13px;
      color: #c1cfe8;
      font-weight: bold;
      box-shadow: 0 0 25px rgba(${hexToRgb(colorHex)}, 0.6);
      z-index: 18;
      white-space: nowrap;
      pointer-events: none;
      user-select: none;
      animation: depthLabelFadeIn 0.4s ease-out;
    `;
    
    label.innerHTML = `
      <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 18px;">${category.emoji || '📏'}</span>
        <div>
          <div style="color: ${colorCode}; font-size: 14px; font-weight: bold; letter-spacing: 0.5px;">
            TUMOR DEPTH
          </div>
          <div style="color: #5a7a99; font-size: 12px; margin-top: 3px;">
            ${depth.toFixed(1)} mm from cortex
          </div>
          <div style="color: ${colorCode}; font-size: 11px; margin-top: 4px; text-transform: uppercase;">
            ${category.label}
          </div>
        </div>
      </div>
    `;
    
    document.body.appendChild(label);
    
    // Store for position updates
    window.depthLabelElement = label;
    window.depthLabelWorldPos = worldPos;
    
    console.log('[Brain3D] ✅ Depth label created');
  }
  function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec('#' + hex.toString(16).padStart(6, '0'));
    if (!result) return '255, 0, 0';
    return [
      parseInt(result[1], 16),
      parseInt(result[2], 16),
      parseInt(result[3], 16)
    ].join(', ');
  }

  //============================================
 // ===== BUILD 3D COORDINATE MARKERS (NEW) =====
function buildCoordinateMarkers(tumorPoints, metrics) {
  console.log(`%c[Brain3D] 🎯 Building 3D coordinate markers`, 'color: #00c853; font-weight: bold;');
  
  if (!tumorPoints || tumorPoints.length === 0 || !metrics) return;
  
  // Remove old markers
  if (window.coordinateMarkers) {
    window.coordinateMarkers.forEach(marker => scene.remove(marker));
    window.coordinateMarkers = [];
  } else {
    window.coordinateMarkers = [];
  }
  
  const centroid = metrics.centroid_mm;
  if (!centroid) return;
  
  // Normalize centroid to 3D space (-1 to 1)
  const scale = 1.0;
  const centerX = (centroid[0] / 55) * scale;  // 55mm ≈ brain radius
  const centerY = (centroid[1] / 55) * scale;
  const centerZ = (centroid[2] / 55) * scale;
  
  // 1. CENTER SPHERE (Tumor centroid)
  const sphereGeom = new THREE.SphereGeometry(0.08, 16, 16);
  const sphereMat = new THREE.MeshBasicMaterial({
    color: 0xFF0040,
    emissive: 0xFF0040,
    emissiveIntensity: 0.8
  });
  const centerMarker = new THREE.Mesh(sphereGeom, sphereMat);
  centerMarker.position.set(centerX, centerY, centerZ);
  
  // Apply rotation
  centerMarker.rotation.x = currentRotX;
  centerMarker.rotation.y = currentRotY;
  
  scene.add(centerMarker);
  window.coordinateMarkers.push(centerMarker);
  
  // 2. AXIS LINES (X, Y, Z)
  const lineLength = 0.3;
  const lineWidth = 2;
  
  // X axis (red)
  const xGeom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(centerX - lineLength/2, centerY, centerZ),
    new THREE.Vector3(centerX + lineLength/2, centerY, centerZ)
  ]);
  const xLine = new THREE.Line(xGeom, new THREE.LineBasicMaterial({ color: 0xFF5555, linewidth: lineWidth }));
  xLine.rotation.x = currentRotX;
  xLine.rotation.y = currentRotY;
  scene.add(xLine);
  window.coordinateMarkers.push(xLine);
  
  // Y axis (green)
  const yGeom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(centerX, centerY - lineLength/2, centerZ),
    new THREE.Vector3(centerX, centerY + lineLength/2, centerZ)
  ]);
  const yLine = new THREE.Line(yGeom, new THREE.LineBasicMaterial({ color: 0x55FF55, linewidth: lineWidth }));
  yLine.rotation.x = currentRotX;
  yLine.rotation.y = currentRotY;
  scene.add(yLine);
  window.coordinateMarkers.push(yLine);
  
  // Z axis (blue)
  const zGeom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(centerX, centerY, centerZ - lineLength/2),
    new THREE.Vector3(centerX, centerY, centerZ + lineLength/2)
  ]);
  const zLine = new THREE.Line(zGeom, new THREE.LineBasicMaterial({ color: 0x5555FF, linewidth: lineWidth }));
  zLine.rotation.x = currentRotX;
  zLine.rotation.y = currentRotY;
  scene.add(zLine);
  window.coordinateMarkers.push(zLine);
  
  // 3. DISTANCE INDICATOR (to cortex)
  if (metrics.distance_to_cortex_mm) {
    const distanceGeom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(1.05 * centerX, 1.05 * centerY, 1.05 * centerZ),
      new THREE.Vector3(1.1 * centerX, 1.1 * centerY, 1.1 * centerZ)
    ]);
    const distanceLine = new THREE.Line(
      distanceGeom,
      new THREE.LineBasicMaterial({ color: 0x9c27b0, linewidth: 2 })
    );
    distanceLine.rotation.x = currentRotX;
    distanceLine.rotation.y = currentRotY;
    scene.add(distanceLine);
    window.coordinateMarkers.push(distanceLine);
  }
  
  console.log('[Brain3D] ✅ Coordinate markers created:', window.coordinateMarkers.length);
}

// Gọi hàm này từ window.updateBrainTumor:
window.updateBrainTumor = function(tumorPoints, metrics, depthMetrics) {
  if (!tumorPoints || tumorPoints.length === 0) return;
  
  console.log(`%c[Brain3D] 🎯 UPDATING TUMOR with depth metrics`, 'color: #FF1744; font-weight: bold; font-size: 14px;');
  
  currentTumorPoints = tumorPoints;
  lastTumorHoverData = metrics;
  window.lastDepthMetrics = depthMetrics;
  
  highlightTumorRegion(tumorPoints);
  buildTumorParticles(tumorPoints);
  buildTumorSpikes(tumorPoints);
  buildTumorContour(tumorPoints);
  buildCoordinateMarkers(tumorPoints, metrics);
  
  // ✅ FIX: Truyền depthMetrics đúng
  if (depthMetrics) {
    buildTumorDepthVector(depthMetrics);
  } else {
    console.warn('[Brain3D] ⚠️  No depth metrics provided');
  }
};

  // ---- Detail View Toggle - SIMPLIFIED (NO SLICE CUTTING) ----
  window.toggleDetailView = function() {
    isDetailView = !isDetailView;

    const indicator = document.getElementById('sliceIndicator');
    if (indicator) indicator.style.display = isDetailView ? 'block' : 'none';

    if (isDetailView) {
      console.log('%c[Brain3D] 🔍 Detail View ENABLED - Switching to detail_brain', 'color: #00ffff; font-weight: bold;');
      switchBrainModel('detail');
    } else {
      console.log('%c[Brain3D] 🔍 Detail View DISABLED - Switching to normal brain', 'color: #00ffff; font-weight: bold;');
      switchBrainModel('normal');
    }

    return isDetailView;
  };

  // Keep old function name for compatibility
  window.toggleSliceView = function() {
    return window.toggleDetailView();
  };

  // ---- Reset View ----
  window.resetBrainView = function() {
    targetRotX = 0;
    targetRotY = 0;
    isAutoRotating = true;
    
    camera.position.set(0, 0.8, 5.0);
    camera.lookAt(0, 0, 0);
    
    // If detail view is active, turn it off
    if (isDetailView) {
      isDetailView = false;
      switchBrainModel('normal');
      
      const indicator = document.getElementById('sliceIndicator');
      if (indicator) indicator.style.display = 'none';
    }
  };

  // ---- Toggle Auto-Rotate ----
  window.toggleAutoRotate = function() {
    isAutoRotating = !isAutoRotating;
    return isAutoRotating;
  };

  // ---- Mouse Events ----
  function setupEvents(container) {
    container.addEventListener('mousedown', (e) => {
      isDragging = true;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
    });
    
    container.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      const dx = e.clientX - lastMouseX;
      const dy = e.clientY - lastMouseY;
      targetRotY += dx * 0.008;
      targetRotX += dy * 0.008;
      targetRotX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, targetRotX));
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
    });
    
    container.addEventListener('mouseup', () => { isDragging = false; });
    container.addEventListener('mouseleave', () => { isDragging = false; });

    // Touch support
    container.addEventListener('touchstart', (e) => {
      isDragging = true;
      lastMouseX = e.touches[0].clientX;
      lastMouseY = e.touches[0].clientY;
    });
    
    container.addEventListener('touchmove', (e) => {
      if (!isDragging) return;
      const dx = e.touches[0].clientX - lastMouseX;
      const dy = e.touches[0].clientY - lastMouseY;
      targetRotY += dx * 0.008;
      targetRotX += dy * 0.008;
      lastMouseX = e.touches[0].clientX;
      lastMouseY = e.touches[0].clientY;
    });
    
     // ✅ THÊM MOUSE MOVE EVENT:
    container.addEventListener('mousemove', (e) => {
      if (isDragging) return;  // Don't show tooltip while dragging
      detectTumorHover(e);
    });
    
    // ✅ THÊM CLICK EVENT:
    container.addEventListener('click', (e) => {
      handleTumorClick(e);
    });
    
    // ✅ THÊM MOUSE LEAVE EVENT:
    container.addEventListener('mouseleave', () => {
      hideTumorTooltip();
    });
    

    container.addEventListener('touchend', () => { isDragging = false; });

    // Zoom
    container.addEventListener('wheel', (e) => {
      e.preventDefault();
      camera.position.z += e.deltaY * 0.004;
      camera.position.z = Math.max(2.5, Math.min(9.0, camera.position.z));
    }, { passive: false });
  }

  // ---- Animation Loop ----
  function animate() {
    animationId = requestAnimationFrame(animate);
    const dt = clock.getDelta();
    const time = clock.getElapsedTime();
  
    // Auto-rotate
    if (isAutoRotating && !isDragging) {
      targetRotY += dt * 0.25;
    }
  
    // Smooth interpolation
    currentRotX += (targetRotX - currentRotX) * 0.08;
    currentRotY += (targetRotY - currentRotY) * 0.08;
  
    if (brainMesh) {
      brainMesh.rotation.x = currentRotX;
      brainMesh.rotation.y = currentRotY;
    }
    
    // Animate tumor particles
    if (tumorParticles) {
      tumorParticles.rotation.x = currentRotX;
      tumorParticles.rotation.y = currentRotY;
      
      if (tumorParticles.material.uniforms) {
        tumorParticles.material.uniforms.time.value = time;
      }
    }
    
    if (tumorGlow) {
      tumorGlow.rotation.x = currentRotX;
      tumorGlow.rotation.y = currentRotY;
      // AGGRESSIVE GLOW PULSING
      tumorGlow.material.opacity = 0.25 + 0.25 * Math.sin(time * 2.5);
    }
  
    if (tumorSpikes) {
      tumorSpikes.rotation.x = currentRotX;
      tumorSpikes.rotation.y = currentRotY;
    }
  
    if (tumorContour) {
      tumorContour.rotation.x = currentRotX;
      tumorContour.rotation.y = currentRotY;
    }
    if (depthVectorGroup) {
      depthVectorGroup.rotation.x = currentRotX;
      depthVectorGroup.rotation.y = currentRotY;
      animateDepthVector(time);
    }
    
    updateDepthLabelPosition();
  
    renderer.render(scene, camera);
  }

  // ===== BUILD TUMOR DEPTH VECTOR (NEW) =====




// ===== UPDATE DEPTH LABEL POSITION (in animate loop) =====
function updateDepthLabelPosition() {
  if (!window.depthLabelElement || !window.depthLabelWorldPos || !camera) return;
  
  const vector = window.depthLabelWorldPos.clone();
  vector.project(camera);
  
  const x = (vector.x * 0.5 + 0.5) * window.innerWidth;
  const y = -(vector.y * 0.5 - 0.5) * window.innerHeight;
  
  window.depthLabelElement.style.left = (x + 25) + 'px';
  window.depthLabelElement.style.top = (y - 15) + 'px';
}

// ===== ANIMATE DEPTH VECTOR PULSING =====
function animateDepthVector(time) {
  if (!window.depthVectorPulsing) return;
  
  // Pulsing effect
  const pulse = 1 + 0.25 * Math.sin(time * 3);
  const opacityPulse = 0.7 + 0.25 * Math.sin(time * 2.5);
  
  if (window.depthVectorPulsing.line) {
    window.depthVectorPulsing.line.material.opacity = opacityPulse;
  }
  
  if (window.depthVectorPulsing.glow) {
    window.depthVectorPulsing.glow.material.opacity = 0.15 + 0.1 * Math.sin(time * 2.5);
  }
  
  if (window.depthVectorPulsing.arrow) {
    window.depthVectorPulsing.arrow.scale.set(pulse, pulse, pulse);
  }
  
  if (window.depthVectorPulsing.tumorMarker) {
    window.depthVectorPulsing.tumorMarker.scale.set(pulse * 1.2, pulse * 1.2, pulse * 1.2);
  }
}

})();