/**
 * app.js (COMPLETE VERSION WITH XAI INTEGRATION)
 * Main application logic for NeuroScan AI frontend.
 *
 * Features:
 *   - File upload (drag & drop + click)
 *   - MRI preview canvas rendering
 *   - POST to /api/diagnose
 *   - Render segmentation mask overlay
 *   - Display Groq AI diagnosis report
 *   - Trigger 3D brain tumor update
 *   - Tab navigation (Scan, 3D Brain, XAI, Similar Cases, Info)
 *   - XAI Dashboard rendering (Grad-CAM, Rules, SHAP, Insights)
 *   - Similar Cases grid with modal details
 *   - Health check on load
 */

(function App() {

  const API_BASE = 'http://localhost:8000/api';

  // ===== DOM References - Main Report Section =====
  const uploadZone       = document.getElementById('uploadZone');
  const fileInput        = document.getElementById('fileInput');
  const previewWrap      = document.getElementById('previewWrap');
  const previewCanvas    = document.getElementById('previewCanvas');
  const btnDiagnose      = document.getElementById('btnDiagnose');
  const reportPlaceholder = document.getElementById('reportPlaceholder');
  const reportContent    = document.getElementById('reportContent');
  const loadingState     = document.getElementById('loadingState');
  const statusDot        = document.getElementById('statusDot');
  const statusText       = document.getElementById('statusText');

  // ===== Report Fields =====
  const confidenceBar   = document.getElementById('confidenceBar');
  const confidenceValue = document.getElementById('confidenceValue');
  const statStatus      = document.getElementById('statStatus');
  const statArea        = document.getElementById('statArea');
  const statLocation    = document.getElementById('statLocation');
  const statSeverity    = document.getElementById('statSeverity');
  const reportSummary   = document.getElementById('reportSummary');
  const findingsList    = document.getElementById('findingsList');
  const recommendationsList = document.getElementById('recommendationsList');
  const disclaimer      = document.getElementById('disclaimer');

  // ===== Viewer Controls =====
  const btnRotate = document.getElementById('btnRotate');
  const btnSlice  = document.getElementById('btnSlice');
  const btnReset  = document.getElementById('btnReset');

  // ===== Tab Navigation =====
  const pills    = document.querySelectorAll('.pill');
  const infoPanel = document.getElementById('infoPanel');
  const xaiPanel = document.getElementById('xaiPanel');
  const similarPanel = document.getElementById('similarPanel');
  const mainLayout = document.querySelector('.main-layout');

  // ===== State =====
  let currentFile = null;
  let currentImageFile = null;
  let maskOverlayCanvas = null;
  let lastPredictionData = null;
  let lastXAIData = null;
  let lastSimilarData = null;
  let isProcessingDiagnosis = false;

  // ===== HEALTH CHECK =====
  async function checkHealth() {
    try {
      console.log('[App] 🏥 Checking backend health...');
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const health = await res.json();
        console.log('[App] ✅ Backend online:', health);
        statusDot.className  = 'status-dot online';
        statusText.textContent = 'Backend Online';
      } else {
        throw new Error('not ok');
      }
    } catch (err) {
      console.error('[App] ❌ Backend offline:', err);
      statusDot.className  = 'status-dot error';
      statusText.textContent = 'Backend Offline';
    }
  }

  // ===== FILE UPLOAD HANDLERS =====
  uploadZone.addEventListener('click', () => fileInput.click());
  
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
      handleFile(e.target.files[0]);
    }
  });

  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });
  
  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
  });
  
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  function handleFile(file) {
    if (!file) return;
    
    if (!['image/png', 'image/jpeg'].includes(file.type)) {
      alert('❌ Please upload a PNG or JPG image.');
      return;
    }
    
    console.log('[App] 📂 File selected:', file.name, `(${(file.size/1024/1024).toFixed(2)} MB)`);
    
    currentFile = file;
    currentImageFile = file;
    renderPreview(file);
    btnDiagnose.disabled = false;
  }

  // ===== RENDER PREVIEW CANVAS =====
  function renderPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        console.log('[App] 🖼️ Rendering preview...');
        
        // Show preview container
        previewWrap.style.display = 'block';

        // Draw on canvas
        const ctx = previewCanvas.getContext('2d');
        previewCanvas.width  = 256;
        previewCanvas.height = 256;
        ctx.drawImage(img, 0, 0, 256, 256);

        // Remove old mask overlay if exists
        if (maskOverlayCanvas) {
          maskOverlayCanvas.remove();
          maskOverlayCanvas = null;
        }
        
        console.log('[App] ✅ Preview rendered');
      };
      img.onerror = () => {
        alert('❌ Failed to load image');
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  // ===== RENDER MASK OVERLAY =====
  function renderMaskOverlay(mask) {
    // mask is 256×256 array of 0/1 values
    console.log('[App] 🎨 Rendering mask overlay...');
    
    if (maskOverlayCanvas) {
      maskOverlayCanvas.remove();
    }

    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.pointerEvents = 'none';
    canvas.style.borderRadius = '10px';

    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(256, 256);

    for (let y = 0; y < 256; y++) {
      for (let x = 0; x < 256; x++) {
        const idx = (y * 256 + x) * 4;
        const val = mask[y][x];
        
        if (val > 0.5) {
          // Red tumor overlay with alpha
          imageData.data[idx]     = 255;   // R
          imageData.data[idx + 1] = 82;    // G
          imageData.data[idx + 2] = 82;    // B
          imageData.data[idx + 3] = 140;   // A (semi-transparent)
        } else {
          imageData.data[idx + 3] = 0;     // transparent
        }
      }
    }

    ctx.putImageData(imageData, 0, 0);
    previewWrap.appendChild(canvas);
    maskOverlayCanvas = canvas;
    
    console.log('[App] ✅ Mask overlay rendered');
  }

  // ===== RUN DIAGNOSIS =====
 // ===== RUN DIAGNOSIS (UPDATED) =====
btnDiagnose.addEventListener('click', async () => {
  if (!currentFile || isProcessingDiagnosis) return;

  isProcessingDiagnosis = true;
  console.log('%c[App] 🚀 Starting diagnosis...', 'color: #00e5ff; font-weight: bold;');
  
  showState('loading');
  btnDiagnose.disabled = true;

  try {
    // 1️⃣ Call main diagnosis API
    const formData = new FormData();
    formData.append('file', currentFile);

    console.log('[App] 📡 Sending to /api/diagnose...');
    const res = await fetch(`${API_BASE}/diagnose`, {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Diagnosis failed');
    }

    const diagnosisResult = await res.json();
    console.log('[App] ✅ Diagnosis complete:', diagnosisResult);

    // Store prediction
    lastPredictionData = diagnosisResult.prediction;

    // ✅ Store XAI data (already included in response)
    if (diagnosisResult.xai && !diagnosisResult.xai.error) {
      lastXAIData = diagnosisResult.xai;
      window.lastXAIData = diagnosisResult.xai;
      console.log('[App] ✅ XAI data stored:', lastXAIData);
    } else {
      console.warn('[App] ⚠️  XAI data not available');
      lastXAIData = null;
      window.lastXAIData = null;
    }
    
    // 2️⃣ Render mask overlay
    if (diagnosisResult.mask) {
      renderMaskOverlay(diagnosisResult.mask);
    }
    
    // 3️⃣ Display report
    displayReport(diagnosisResult);
    
    // 4️⃣ Update 3D brain
    update3DBrain(diagnosisResult);
    
    // 5️⃣ Fetch Similar cases (optional, separate call)
    if (currentImageFile && window.XAISimilarUI?.fetchSimilarCases) {
      try {
        const similarData = await window.XAISimilarUI.fetchSimilarCases(currentImageFile);
        lastSimilarData = similarData;
        window.lastSimilarData = similarData;
        console.log('[App] ✅ Similar cases received');
      } catch (err) {
        console.warn('[App] ⚠️  Similar fetch failed:', err);
      }
    }
    
    // Show report
    showState('report');

  } catch (err) {
    console.error('[App] ❌ Diagnosis error:', err);
    alert('❌ Error: ' + err.message);
    showState('placeholder');
  } finally {
    isProcessingDiagnosis = false;
    btnDiagnose.disabled = false;
  }
});

  // ===== FETCH XAI DATA =====
  async function fetchXAIData(diagnosisResult) {
    try {
        // XAI data ĐÃ CÓ trong response từ /api/diagnose
        if (diagnosisResult.xai && !diagnosisResult.xai.error) {
            lastXAIData = diagnosisResult.xai;
            console.log('[App] ✅ XAI data from diagnosis:', lastXAIData);
        } else {
            console.warn('[App] ⚠️  XAI data not available in diagnosis response');
            lastXAIData = null;
        }
    } catch (err) {
        console.error('[App] ❌ XAI data error:', err);
        lastXAIData = null;
    }
}

  // ===== FETCH SIMILAR CASES =====
  async function fetchSimilarCases(imageFile) {
    if (!window.XAISimilarUI?.fetchSimilarCases) {
      console.warn('[App] ⚠️  Similar UI not ready');
      return;
    }
    
    try {
      const similarData = await window.XAISimilarUI.fetchSimilarCases(imageFile);
      lastSimilarData = similarData;
      console.log('[App] ✅ Similar cases received:', similarData);
    } catch (err) {
      console.warn('[App] ⚠️  Similar fetch failed:', err);
    }
  }

  // ===== DISPLAY REPORT =====
  function displayReport(data) {
    console.log('[App] 📋 Displaying report...');
    
    const pred   = data.prediction;
    const report = data.report;

    // Confidence
    const confPct = Math.round(pred.confidence * 100);
    confidenceBar.style.width = confPct + '%';
    confidenceValue.textContent = confPct + '%';

    // Status
    statStatus.textContent = pred.tumor_detected ? 'Tumor Detected' : 'No Tumor';
    statStatus.className   = 'stat-value ' + (pred.tumor_detected ? 'detected' : 'clear');

    // Area
    statArea.textContent   = pred.tumor_area_percent + '%';

    // Location
    statLocation.textContent = pred.location_hint || 'N/A';

    // Severity
    const sev = report.severity || 'Unknown';
    statSeverity.textContent = sev;
    statSeverity.className   = 'stat-value severity-' + sev.toLowerCase();

    // Summary
    reportSummary.textContent = report.summary || '—';

    // Findings
    findingsList.innerHTML = (report.findings || [])
      .map(f => `<li>✓ ${f}</li>`)
      .join('');

    // Recommendations
    recommendationsList.innerHTML = (report.recommendations || [])
      .map(r => `<li>→ ${r}</li>`)
      .join('');

     // Populate methods comparison table
    const comparisonBody = document.getElementById('methodsComparisonBody');

    if (comparisonBody && data.xai) {
      const methods = [
          {
              name: 'CNN Segmentation',
              result: data.prediction.tumor_detected ? 'Tumor Detected' : 'No Tumor',
              confidence: `${(data.prediction.confidence * 100).toFixed(1)}%`
          },
          {
              name: 'Grad-CAM Attention',
              result: data.xai.gradcam ? `${data.xai.gradcam.confidence_level} Focus` : 'N/A',
              confidence: data.xai.gradcam ? `${(data.xai.gradcam.attention_score * 100).toFixed(1)}%` : 'N/A'
          },
          {
              name: 'Rule-Based Analysis',
              result: data.xai.rule_based ? data.xai.rule_based.risk_level : 'N/A',
              confidence: data.xai.rule_based ? `${data.xai.rule_based.risk_rationale?.risk_score || 0}/9 score` : 'N/A'
          },
          {
              name: 'SHAP Analysis',
              result: data.xai.shap && data.xai.shap.top_features ? 
                  `Top: ${data.xai.shap.top_features[0]}` : 'N/A',
              confidence: data.xai.shap && data.xai.shap.top_features ? 
                  `${(data.xai.shap.feature_importance[data.xai.shap.top_features[0]] * 100).toFixed(0)}%` : 'N/A'
          }
      ];
      
      comparisonBody.innerHTML = methods.map(m => `
          <tr style="border-bottom: 1px solid rgba(30, 58, 82, 0.3);">
              <td style="padding: 10px; color: #c1cfe8;">${m.name}</td>
              <td style="padding: 10px; text-align: center; color: #00e5ff; font-weight: 500;">${m.result}</td>
              <td style="padding: 10px; text-align: center; color: #8899b0;">${m.confidence}</td>
          </tr>
      `).join('');
  }
    // Disclaimer
    disclaimer.textContent = report.disclaimer ||
      '⚠️ This is an AI-generated report. Not a substitute for professional medical advice.';
    
    console.log('[App] ✅ Report displayed');

    
  }

  // ===== UPDATE 3D BRAIN =====
  function update3DBrain(data) {
    if (!data.prediction || !data.prediction.tumor_detected) {
      console.log('[App] ℹ️  No tumor detected, skipping 3D update');
      return;
    }

    console.log('[App] 🧠 Updating 3D brain visualization...');
    
    const location = mapLocationToKey(data.prediction.location_hint);
    const tumorSize = Math.min(data.prediction.tumor_area_percent / 5, 0.5);

    fetch(`${API_BASE}/brain3d?location=${location}&tumor_size=${tumorSize}`)
      .then(r => r.json())
      .then(brainData => {
        if (window.updateBrainTumor && brainData.tumor_points) {
          console.log('[App] ✅ 3D brain updated with tumor points');
          window.updateBrainTumor(brainData.tumor_points);
        }
      })
      .catch(err => console.warn('[App] ⚠️  3D update failed:', err));
  }

  function mapLocationToKey(hint) {
    if (!hint) return 'left_frontal';
    
    const h = hint.toLowerCase();
    
    if (h.includes('left') && h.includes('frontal'))   return 'left_frontal';
    if (h.includes('right') && h.includes('frontal'))  return 'right_frontal';
    if (h.includes('left') && h.includes('temporal'))  return 'left_temporal';
    if (h.includes('right') && h.includes('temporal')) return 'right_temporal';
    if (h.includes('left') && h.includes('parietal'))  return 'left_parietal';
    if (h.includes('right') && h.includes('parietal')) return 'right_parietal';
    if (h.includes('superior') && h.includes('left'))  return 'superior_left';
    if (h.includes('inferior'))                        return 'inferior_right';
    
    return 'left_frontal';
  }

  // ===== STATE MANAGEMENT =====
  function showState(state) {
    reportPlaceholder.style.display = 'none';
    reportContent.style.display     = 'none';
    loadingState.style.display      = 'none';

    switch(state) {
      case 'loading':
        loadingState.style.display = 'flex';
        break;
      case 'report':
        reportContent.style.display = 'block';
        break;
      case 'placeholder':
        reportPlaceholder.style.display = 'flex';
        break;
    }
  }

  // ===== TAB NAVIGATION =====
 // ===== TAB NAVIGATION (FIXED) =====
function switchTab(tabName) {
  console.log(`[App] 📑 Switching to tab: ${tabName}`);
  
  // Hide all panels
  if (mainLayout) mainLayout.style.display = 'none';
  if (xaiPanel) xaiPanel.style.display = 'none';
  if (similarPanel) similarPanel.style.display = 'none';
  if (infoPanel) infoPanel.style.display = 'none';

  // Show based on tab
  switch(tabName) {
    case 'scan':
    case 'brain3d':
      if (mainLayout) mainLayout.style.display = 'grid';
      break;

    case 'xai':
      // ✅ Call XAISimilarUI to render XAI panel
      if (window.XAISimilarUI?.renderXAIDashboard && lastXAIData) {
        window.XAISimilarUI.renderXAIDashboard(lastXAIData);
      } else if (window.XAISimilarUI?.showXAIPanel) {
        window.XAISimilarUI.showXAIPanel();
      } else {
        // Fallback placeholder
        if (xaiPanel) {
          xaiPanel.innerHTML = `
            <div style="padding: 80px 40px; text-align: center; color: #8899b0;">
              <div style="font-size: 64px; margin-bottom: 24px;">🔍</div>
              <h2 style="color: #00e5ff;">XAI Analysis</h2>
              <p style="margin-top: 16px;">
                Upload an MRI image and run diagnosis to see XAI analysis.
              </p>
            </div>
          `;
        }
      }
      if (xaiPanel) xaiPanel.style.display = 'block';
      break;

    case 'similar':
      // ✅ Call XAISimilarUI to render Similar panel
      if (window.XAISimilarUI?.renderSimilarCases && lastSimilarData) {
        window.XAISimilarUI.renderSimilarCases(lastSimilarData);
      } else if (window.XAISimilarUI?.showSimilarPanel) {
        window.XAISimilarUI.showSimilarPanel();
      } else {
        // Fallback placeholder
        if (similarPanel) {
          similarPanel.innerHTML = `
            <div style="padding: 80px 40px; text-align: center; color: #8899b0;">
              <div style="font-size: 64px; margin-bottom: 24px;">🔎</div>
              <h2 style="color: #00e5ff;">Similar Cases</h2>
              <p style="margin-top: 16px;">
                Upload an MRI to find similar cases from database.
              </p>
            </div>
          `;
        }
      }
      if (similarPanel) similarPanel.style.display = 'block';
      break;

    case 'info':
      if (infoPanel) infoPanel.style.display = 'block';
      break;
  }

  // Update active pill
  pills.forEach(p => p.classList.remove('active'));
  document.querySelector(`.pill[data-tab="${tabName}"]`)?.classList.add('active');
}

  // ===== RENDER XAI PANEL (FIXED) =====
  function renderXAIPanel() {
    if (!xaiPanel) {
      console.error('[App] XAI panel element not found');
      return;
    }
    
    if (!lastXAIData) {
      console.log('[App] No XAI data yet, showing placeholder');
      xaiPanel.innerHTML = `
        <div style="padding: 80px 40px; text-align: center; color: #8899b0; max-width: 600px; margin: 0 auto;">
          <div style="font-size: 64px; margin-bottom: 24px;">🔍</div>
          <h2 style="color: #00e5ff; font-size: 24px; margin-bottom: 16px;">XAI Analysis</h2>
          <p style="font-size: 16px; line-height: 1.6; margin: 0;">
            Upload an MRI image and run diagnosis to see explainable AI analysis here.
          </p>
          <p style="font-size: 14px; margin-top: 12px; color: #4a5568;">
            The system will provide Grad-CAM heatmaps, rule-based analysis, SHAP feature importance, and combined insights.
          </p>
        </div>
      `;
      return;
    }
    
    // Call XAISimilarUI module to render dashboard
    console.log('[App] Rendering XAI dashboard with data:', lastXAIData);
    
    if (window.XAISimilarUI?.renderXAIDashboard) {
      window.XAISimilarUI.renderXAIDashboard(lastXAIData);
      console.log('[App] ✅ XAI dashboard rendered');
    } else {
      console.error('[App] XAISimilarUI.renderXAIDashboard not available');
      xaiPanel.innerHTML = `
        <div style="padding: 40px; text-align: center; color: #ff5252;">
          <p>⚠️ XAI rendering module not loaded</p>
          <p style="font-size: 12px; margin-top: 8px;">Check that xai_similar_ui.js is loaded before app.js</p>
        </div>
      `;
    }
  }

  // ===== RENDER SIMILAR PANEL (FIXED) =====
  function renderSimilarPanel() {
    if (!similarPanel) {
      console.error('[App] Similar panel element not found');
      return;
    }
    
    if (!lastSimilarData) {
      console.log('[App] No similar data yet, showing placeholder');
      similarPanel.innerHTML = `
        <div style="padding: 80px 40px; text-align: center; color: #8899b0; max-width: 600px; margin: 0 auto;">
          <div style="font-size: 64px; margin-bottom: 24px;">🔎</div>
          <h2 style="color: #00e5ff; font-size: 24px; margin-bottom: 16px;">Similar Cases</h2>
          <p style="font-size: 16px; line-height: 1.6; margin: 0;">
            Upload an MRI image to find similar cases from the database.
          </p>
          <p style="font-size: 14px; margin-top: 12px; color: #4a5568;">
            The system will search for visually similar brain scans and their diagnosis results.
          </p>
        </div>
      `;
      return;
    }
    
    // Call XAISimilarUI module to render similar cases
    console.log('[App] Rendering similar cases with data:', lastSimilarData);
    
    if (window.XAISimilarUI?.renderSimilarCases) {
      window.XAISimilarUI.renderSimilarCases(lastSimilarData);
      console.log('[App] ✅ Similar cases rendered');
    } else {
      console.error('[App] XAISimilarUI.renderSimilarCases not available');
      similarPanel.innerHTML = `
        <div style="padding: 40px; text-align: center; color: #ff5252;">
          <p>⚠️ Similar cases rendering module not loaded</p>
          <p style="font-size: 12px; margin-top: 8px;">Check that xai_similar_ui.js is loaded before app.js</p>
        </div>
      `;
    }
  }

  // Attach tab click handlers
  pills.forEach(pill => {
    pill.addEventListener('click', (e) => {
      const tab = e.target.dataset.tab;
      switchTab(tab);
    });
  });

  // ===== VIEWER CONTROLS =====
  if (btnRotate) {
    btnRotate.addEventListener('click', () => {
      const active = window.toggleAutoRotate && window.toggleAutoRotate();
      btnRotate.classList.toggle('active', active);
      console.log('[App] 🔄 Auto-rotate:', active ? 'ON' : 'OFF');
    });
  }

  if (btnSlice) {
    btnSlice.addEventListener('click', () => {
      const active = window.toggleDetailView && window.toggleDetailView();
      btnSlice.classList.toggle('active', active);
      console.log('[App] 🔍 Detail view:', active ? 'ON' : 'OFF');
    });
  }

  if (btnReset) {
    btnReset.addEventListener('click', () => {
      if (window.resetBrainView) {
        window.resetBrainView();
      }
      btnRotate.classList.add('active');
      btnSlice.classList.remove('active');
      console.log('[App] ↺ Brain view reset');
    });
  }

  // ===== INITIALIZATION =====
  window.addEventListener('DOMContentLoaded', () => {
    console.log('%c[App] 🚀 Initializing NeuroScan AI...', 'color: #00e5ff; font-weight: bold; font-size: 14px;');
    
    // Health check
    checkHealth();
    
    // Init 3D viewer (defined in brain3d.js)
    if (window.initBrainViewer) {
      window.initBrainViewer();
      console.log('[App] ✅ 3D viewer initialized');
    }
    
    // Init XAI/Similar UI (defined in xai_similar_ui.js)
    if (window.XAISimilarUI?.init) {
      window.XAISimilarUI.init();
      console.log('[App] ✅ XAI/Similar UI initialized');
    } else {
      console.warn('[App] ⚠️  XAISimilarUI not available - check script loading order');
    }
    
    // Set rotate button active by default
    if (btnRotate) btnRotate.classList.add('active');
    
    console.log('%c[App] ✅ All systems ready!', 'color: #00c853; font-weight: bold; font-size: 14px;');
  });

  // ===== WINDOW EXPORTS (for external scripts) =====
  window.App = {
    switchTab,
    lastPredictionData: () => lastPredictionData,
    lastXAIData: () => lastXAIData,
    lastSimilarData: () => lastSimilarData,
    getCurrentFile: () => currentFile
  };

})();