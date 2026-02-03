/**
 * brain3d.js
 * Three.js 3D Brain Visualization Engine
 *
 * Features:
 *   - Procedural brain-shaped mesh (spherical harmonics deformation)
 *   - Tumor region rendered as glowing particle cluster
 *   - Auto-rotation with mouse drag override
 *   - Slice view (cross-section cutting plane)
 *   - Soft lighting for medical visualization feel
 *
 * No external 3D model files needed — everything is generated procedurally.
 */

(function BrainViewer() {

    // ---- State ----
    let scene, camera, renderer, brainMesh, tumorParticles;
    let animationId = null;
    let isAutoRotating = true;
    let isSliceView = false;
    let slicePlane = null;
    let mouseX = 0, mouseY = 0;
    let targetRotX = 0, targetRotY = 0;
    let currentRotX = 0, currentRotY = 0;
    let isDragging = false;
    let lastMouseX = 0, lastMouseY = 0;
    let clock;
  
    // ---- Init ----
    window.initBrainViewer = function() {
      const container = document.getElementById('viewer3d');
      const canvas    = document.getElementById('brainCanvas');
  
      // Scene
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x050810);
  
      // Camera
      const w = container.clientWidth;
      const h = container.clientHeight || 400;
      camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 100);
      camera.position.set(0, 0, 3.8);
  
      // Renderer (software fallback if no GPU)
      renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        antialias: true,
        alpha: true
      });
      renderer.setSize(w, h);
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.shadowMap.enabled = true;
  
      // Lighting
      const ambientLight = new THREE.AmbientLight(0x334466, 0.6);
      scene.add(ambientLight);
  
      const keyLight = new THREE.DirectionalLight(0x88ccee, 0.9);
      keyLight.position.set(2, 3, 2);
      scene.add(keyLight);
  
      const rimLight = new THREE.DirectionalLight(0x00e5ff, 0.3);
      rimLight.position.set(-2, -1, -1);
      scene.add(rimLight);
  
      // Build default brain
      buildBrain();
  
      // Clock for animation
      clock = new THREE.Clock();
  
      // Event listeners
      setupEvents(container);
  
      // Start render loop
      animate();
  
      // Resize handler
      window.addEventListener('resize', () => {
        const w2 = container.clientWidth;
        const h2 = container.clientHeight || 400;
        camera.aspect = w2 / h2;
        camera.updateProjectionMatrix();
        renderer.setSize(w2, h2);
      });
    };
  
    // ---- Build Brain Mesh ----
    function buildBrain(tumorData) {
      // Remove old
      if (brainMesh) { scene.remove(brainMesh); brainMesh.geometry.dispose(); }
      if (tumorParticles) { scene.remove(tumorParticles); tumorParticles.geometry.dispose(); }
  
      const resolution = 64;
      const geometry   = new THREE.BufferGeometry();
      const vertices   = [];
      const colors     = [];
  
      // Brain color palette
      const healthyColor = new THREE.Color(0x1a6b8a);  // teal
      const tumorColor   = new THREE.Color(0xff5252);   // red
  
      // Generate brain-shaped sphere using spherical harmonics
      const phiSteps   = resolution;
      const thetaSteps = resolution;
  
      for (let i = 0; i <= phiSteps; i++) {
        const phi = Math.PI * i / phiSteps;
        for (let j = 0; j <= thetaSteps; j++) {
          const theta = 2 * Math.PI * j / thetaSteps;
  
          // Base radius with brain deformation
          let r = 1.0;
          r += 0.12 * Math.sin(2 * phi) * Math.cos(theta);
          r += 0.08 * Math.sin(phi) * Math.cos(2 * theta);
          r += 0.05 * Math.sin(3 * phi) * Math.sin(theta);
          r += 0.06 * Math.cos(2 * phi);
          // Hemisphere split (sulcus)
          r -= 0.02 * Math.abs(Math.cos(theta));
  
          const x = r * Math.sin(phi) * Math.cos(theta);
          const y = r * Math.cos(phi);
          const z = r * Math.sin(phi) * Math.sin(theta);
  
          vertices.push(x, y, z);
  
          // Default healthy color
          colors.push(healthyColor.r, healthyColor.g, healthyColor.b);
        }
      }
  
      // Generate triangle indices
      const indices = [];
      for (let i = 0; i < phiSteps; i++) {
        for (let j = 0; j < thetaSteps; j++) {
          const a = i * (thetaSteps + 1) + j;
          const b = a + 1;
          const c = (i + 1) * (thetaSteps + 1) + j;
          const d = c + 1;
  
          indices.push(a, c, b);
          indices.push(b, c, d);
        }
      }
  
      geometry.setIndex(indices);
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
      geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
      geometry.computeVertexNormals();
  
      // Material
      const material = new THREE.MeshPhongMaterial({
        vertexColors: true,
        shininess: 40,
        specular: new THREE.Color(0x222244),
        transparent: true,
        opacity: isSliceView ? 0.45 : 0.92
      });
  
      brainMesh = new THREE.Mesh(geometry, material);
      scene.add(brainMesh);
  
      // Apply tumor coloring if data provided
      if (tumorData && tumorData.length > 0) {
        colorTumorRegion(tumorData);
        buildTumorParticles(tumorData);
      }
    }
  
    // ---- Color tumor region on brain surface ----
    function colorTumorRegion(tumorPoints) {
      if (!brainMesh) return;
  
      const positions = brainMesh.geometry.attributes.position;
      const colorsArr = brainMesh.geometry.attributes.color;
      const tumorColor = new THREE.Color(0xff5252);
      const edgeColor  = new THREE.Color(0xff9100);
      const count = positions.count;
  
      // For each vertex, check distance to any tumor point
      for (let i = 0; i < count; i++) {
        const vx = positions.getX(i);
        const vy = positions.getY(i);
        const vz = positions.getZ(i);
  
        let minDist = Infinity;
        for (const tp of tumorPoints) {
          const dx = vx - tp[0], dy = vy - tp[1], dz = vz - tp[2];
          const d = Math.sqrt(dx*dx + dy*dy + dz*dz);
          if (d < minDist) minDist = d;
        }
  
        if (minDist < 0.18) {
          // Core tumor
          colorsArr.setXYZ(i, tumorColor.r, tumorColor.g, tumorColor.b);
        } else if (minDist < 0.28) {
          // Tumor edge (orange gradient)
          const t = (minDist - 0.18) / 0.1;
          colorsArr.setXYZ(i,
            edgeColor.r * (1-t) + 0.1044 * t,
            edgeColor.g * (1-t) + 0.4196 * t,
            edgeColor.b * (1-t) + 0.5451 * t
          );
        }
      }
      colorsArr.needsUpdate = true;
    }
  
    // ---- Build glowing tumor particle cluster ----
    function buildTumorParticles(tumorPoints) {
      if (tumorParticles) { scene.remove(tumorParticles); tumorParticles.geometry.dispose(); }
  
      const geo = new THREE.BufferGeometry();
      const pos = [];
  
      for (const tp of tumorPoints) {
        // Scatter slightly inside the surface
        const scale = 0.95 + Math.random() * 0.08;
        pos.push(tp[0] * scale, tp[1] * scale, tp[2] * scale);
      }
  
      geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  
      const mat = new THREE.PointsMaterial({
        color: 0xff5252,
        size: 0.025,
        transparent: true,
        opacity: 0.7,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      });
  
      tumorParticles = new THREE.Points(geo, mat);
      scene.add(tumorParticles);
    }
  
    // ---- Public: Update tumor on 3D model ----
    window.updateBrainTumor = function(tumorPoints) {
      // Rebuild brain with new tumor
      buildBrain(tumorPoints);
    };
  
    // ---- Slice View Toggle ----
    window.toggleSliceView = function() {
      isSliceView = !isSliceView;
  
      if (brainMesh) {
        brainMesh.material.opacity = isSliceView ? 0.35 : 0.92;
        brainMesh.material.needsUpdate = true;
      }
  
      // Show/hide slice indicator
      const indicator = document.getElementById('sliceIndicator');
      if (indicator) indicator.style.display = isSliceView ? 'block' : 'none';
  
      // Add/remove clipping plane
      if (isSliceView) {
        renderer.localClippingPlanes = [new THREE.Plane(new THREE.Vector3(0, -1, 0), 0.05)];
        if (brainMesh) brainMesh.material.clippingPlanes = renderer.localClippingPlanes;
      } else {
        renderer.localClippingPlanes = [];
        if (brainMesh) brainMesh.material.clippingPlanes = [];
      }
  
      return isSliceView;
    };
  
    // ---- Reset View ----
    window.resetBrainView = function() {
      targetRotX = 0;
      targetRotY = 0;
      isAutoRotating = true;
      isSliceView = false;
      if (brainMesh) {
        brainMesh.material.opacity = 0.92;
        brainMesh.material.clippingPlanes = [];
      }
      renderer.localClippingPlanes = [];
      const indicator = document.getElementById('sliceIndicator');
      if (indicator) indicator.style.display = 'none';
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
      container.addEventListener('touchend', () => { isDragging = false; });
  
      // Scroll to zoom
      container.addEventListener('wheel', (e) => {
        e.preventDefault();
        camera.position.z += e.deltaY * 0.003;
        camera.position.z = Math.max(2.0, Math.min(6.0, camera.position.z));
      }, { passive: false });
    }
  
    // ---- Animation Loop ----
    function animate() {
      animationId = requestAnimationFrame(animate);
      const dt = clock.getDelta();
  
      // Auto-rotate
      if (isAutoRotating && !isDragging) {
        targetRotY += dt * 0.4;
      }
  
      // Smooth interpolation
      currentRotX += (targetRotX - currentRotX) * 0.08;
      currentRotY += (targetRotY - currentRotY) * 0.08;
  
      if (brainMesh) {
        brainMesh.rotation.x = currentRotX;
        brainMesh.rotation.y = currentRotY;
      }
      if (tumorParticles) {
        tumorParticles.rotation.x = currentRotX;
        tumorParticles.rotation.y = currentRotY;
        // Subtle pulsing
        const pulse = 1.0 + 0.15 * Math.sin(Date.now() * 0.003);
        tumorParticles.material.opacity = 0.5 + 0.25 * Math.sin(Date.now() * 0.004);
      }
  
      // Slice plane animation (oscillate if in slice mode)
      if (isSliceView && renderer.localClippingPlanes.length > 0) {
        const sliceY = 0.05 + 0.3 * Math.sin(Date.now() * 0.001);
        renderer.localClippingPlanes[0].constant = sliceY;
      }
  
      renderer.render(scene, camera);
    }
  
  })();