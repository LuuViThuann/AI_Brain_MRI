/**
 * app.js
 * Main application logic for NeuroScan AI frontend.
 *
 * Responsibilities:
 *   - File upload (drag & drop + click)
 *   - MRI preview canvas rendering
 *   - POST to /api/diagnose
 *   - Render segmentation mask overlay
 *   - Display Groq AI diagnosis report
 *   - Trigger 3D brain tumor update
 *   - Tab navigation
 *   - Health check on load
 */

(function App() {

    const API_BASE = 'http://localhost:8000/api';
  
    // ---- DOM References ----
    const uploadZone     = document.getElementById('uploadZone');
    const fileInput      = document.getElementById('fileInput');
    const previewWrap    = document.getElementById('previewWrap');
    const previewCanvas  = document.getElementById('previewCanvas');
    const btnDiagnose    = document.getElementById('btnDiagnose');
    const reportPlaceholder = document.getElementById('reportPlaceholder');
    const reportContent  = document.getElementById('reportContent');
    const loadingState   = document.getElementById('loadingState');
    const statusDot      = document.getElementById('statusDot');
    const statusText     = document.getElementById('statusText');
  
    // Report fields
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
  
    // Viewer controls
    const btnRotate = document.getElementById('btnRotate');
    const btnSlice  = document.getElementById('btnSlice');
    const btnReset  = document.getElementById('btnReset');
  
    // Tab pills
    const pills    = document.querySelectorAll('.pill');
    const infoPanel = document.getElementById('infoPanel');
  
    // ---- State ----
    let currentFile = null;
    let maskOverlayCanvas = null;
  
    // ---- Health Check ----
    async function checkHealth() {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
          statusDot.className  = 'status-dot online';
          statusText.textContent = 'Backend Online';
        } else {
          throw new Error('not ok');
        }
      } catch {
        statusDot.className  = 'status-dot error';
        statusText.textContent = 'Backend Offline';
      }
    }
  
    // ---- File Upload Handlers ----
    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
  
    uploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadZone.classList.add('drag-over');
    });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
    uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadZone.classList.remove('drag-over');
      if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
  
    function handleFile(file) {
      if (!file) return;
      if (!['image/png', 'image/jpeg'].includes(file.type)) {
        alert('Please upload a PNG or JPG image.');
        return;
      }
      currentFile = file;
      renderPreview(file);
      btnDiagnose.disabled = false;
    }
  
    // ---- Render Preview ----
    function renderPreview(file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          // Show preview
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
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(file);
    }
  
    // ---- Render Mask Overlay ----
    function renderMaskOverlay(mask) {
      // mask is 256×256 array of 0/1 values
      // Create an overlay canvas on top of the preview
      if (maskOverlayCanvas) maskOverlayCanvas.remove();
  
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
            imageData.data[idx]     = 255;  // R
            imageData.data[idx + 1] = 82;   // G
            imageData.data[idx + 2] = 82;   // B
            imageData.data[idx + 3] = 140;  // A (semi-transparent)
          } else {
            imageData.data[idx + 3] = 0;    // transparent
          }
        }
      }
  
      ctx.putImageData(imageData, 0, 0);
      previewWrap.appendChild(canvas);
      maskOverlayCanvas = canvas;
    }
  
    // ---- Run Diagnosis ----
    btnDiagnose.addEventListener('click', async () => {
      if (!currentFile) return;
  
      // Show loading
      showState('loading');
      btnDiagnose.disabled = true;
  
      try {
        // Upload to backend
        const formData = new FormData();
        formData.append('file', currentFile);
  
        const res = await fetch(`${API_BASE}/diagnose`, {
          method: 'POST',
          body: formData
        });
  
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Diagnosis failed');
        }
  
        const data = await res.json();
  
        // Render mask overlay on preview
        if (data.mask) {
          renderMaskOverlay(data.mask);
        }
  
        // Display report
        displayReport(data);
  
        // Update 3D brain
        update3DBrain(data);
  
        showState('report');
  
      } catch (err) {
        alert('Error: ' + err.message);
        showState('placeholder');
      } finally {
        btnDiagnose.disabled = false;
      }
    });
  
    // ---- Display Report ----
    function displayReport(data) {
      const pred   = data.prediction;
      const report = data.report;
  
      // Confidence
      const confPct = Math.round(pred.confidence * 100);
      confidenceBar.style.width = confPct + '%';
      confidenceValue.textContent = confPct + '%';
  
      // Stats
      statStatus.textContent = pred.tumor_detected ? 'Tumor Detected' : 'No Tumor';
      statStatus.className   = 'stat-value ' + (pred.tumor_detected ? 'detected' : 'clear');
  
      statArea.textContent   = pred.tumor_area_percent + '%';
      statLocation.textContent = pred.location_hint || 'N/A';
  
      const sev = report.severity || 'Unknown';
      statSeverity.textContent = sev;
      statSeverity.className   = 'stat-value severity-' + sev.toLowerCase();
  
      // Summary
      reportSummary.textContent = report.summary || '—';
  
      // Findings
      findingsList.innerHTML = (report.findings || [])
        .map(f => `<li>${f}</li>`)
        .join('');
  
      // Recommendations
      recommendationsList.innerHTML = (report.recommendations || [])
        .map(r => `<li>${r}</li>`)
        .join('');
  
      // Disclaimer
      disclaimer.textContent = report.disclaimer ||
        '⚠ This is an AI-generated report. Not a substitute for professional medical advice.';
    }
  
    // ---- Update 3D Brain ----
    function update3DBrain(data) {
      if (!data.prediction || !data.prediction.tumor_detected) return;
  
      // Fetch 3D brain data with tumor location
      const location = mapLocationToKey(data.prediction.location_hint);
      const tumorSize = Math.min(data.prediction.tumor_area_percent / 5, 0.5);
  
      fetch(`${API_BASE}/brain3d?location=${location}&tumor_size=${tumorSize}`)
        .then(r => r.json())
        .then(brainData => {
          if (window.updateBrainTumor && brainData.tumor_points) {
            window.updateBrainTumor(brainData.tumor_points);
          }
        })
        .catch(err => console.warn('3D update failed:', err));
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
  
    // ---- State Management ----
    function showState(state) {
      reportPlaceholder.style.display = 'none';
      reportContent.style.display     = 'none';
      loadingState.style.display      = 'none';
  
      if (state === 'loading')     loadingState.style.display     = 'flex';
      if (state === 'report')      reportContent.style.display    = 'block';
      if (state === 'placeholder') reportPlaceholder.style.display = 'flex';
    }
  
    // ---- Tab Navigation ----
    pills.forEach(pill => {
      pill.addEventListener('click', () => {
        pills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
  
        const tab = pill.dataset.tab;
        if (tab === 'info') {
          infoPanel.style.display = 'block';
        } else {
          infoPanel.style.display = 'none';
        }
      });
    });
  
    // ---- Viewer Controls ----
    btnRotate.addEventListener('click', () => {
      const active = window.toggleAutoRotate && window.toggleAutoRotate();
      btnRotate.classList.toggle('active', active);
    });
  
    btnSlice.addEventListener('click', () => {
      const active = window.toggleSliceView && window.toggleSliceView();
      btnSlice.classList.toggle('active', active);
    });
  
    btnReset.addEventListener('click', () => {
      if (window.resetBrainView) window.resetBrainView();
      btnRotate.classList.add('active');
      btnSlice.classList.remove('active');
    });
  
    // ---- Init ----
    window.addEventListener('DOMContentLoaded', () => {
      checkHealth();
      // Init 3D viewer (defined in brain3d.js)
      if (window.initBrainViewer) {
        window.initBrainViewer();
      }
      // Set rotate button active by default
      btnRotate.classList.add('active');
    });
  
  })();