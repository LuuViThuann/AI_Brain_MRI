/**
 * xai_similar_ui.js - UPDATED VERSION (Vietnamese Feature Names)
 * Hiển thị tên feature bằng tiếng Việt
 */

(function XAISimilarUIModule() {
  'use strict';
  
  const API_BASE = 'http://localhost:8000/api';
  const DEBUG = true;
  
  // ===== 🇻🇳 FEATURE NAME TRANSLATION MAP =====
  const FEATURE_NAMES_VI = {
    // Geometric features
    'tumor_area': 'Diện tích khối u',
    'tumor_perimeter': 'Chu vi khối u',
    'circularity': 'Độ tròn',
    'solidity': 'Độ đặc',
    'aspect_ratio': 'Tỷ lệ khung hình',
    'bbox_width': 'Chiều rộng khung',
    'bbox_height': 'Chiều cao khung',
    
    // Location features
    'location_x': 'Vị trí ngang (X)',
    'location_y': 'Vị trí dọc (Y)',
    
    // Intensity features
    'mean_intensity': 'Cường độ trung bình',
    'std_intensity': 'Độ lệch cường độ',
    'min_intensity': 'Cường độ tối thiểu',
    'max_intensity': 'Cường độ tối đa',
    
    // Additional features
    'contrast': 'Độ tương phản',
    'homogeneity': 'Độ đồng nhất',
    'entropy': 'Entropy',
    'edge_density': 'Mật độ cạnh'
  };
  
  // ===== 🇻🇳 FEATURE DESCRIPTIONS (Giải thích chi tiết) =====
  const FEATURE_DESCRIPTIONS_VI = {
    'tumor_area': 'Kích thước vùng khối u (mm²). Yếu tố quan trọng nhất để đánh giá mức độ nghiêm trọng.',
    'tumor_perimeter': 'Chu vi đường viền khối u. Phản ánh hình dạng và ranh giới của khối u.',
    'circularity': 'Mức độ hình tròn của khối u (0-1). Giá trị cao = hình dạng đều đặn.',
    'solidity': 'Tỷ lệ giữa diện tích khối u và diện tích bao lồi. Phản ánh độ đặc của khối u.',
    'aspect_ratio': 'Tỷ lệ chiều rộng/chiều cao. Cho biết khối u có bị kéo dài theo một chiều không.',
    'location_x': 'Vị trí khối u theo chiều ngang (trái-phải). Quan trọng để xác định vùng não bị ảnh hưởng.',
    'location_y': 'Vị trí khối u theo chiều dọc (trên-dưới). Giúp định vị chính xác khối u.',
    'mean_intensity': 'Cường độ sáng trung bình của khối u. Phản ánh mật độ mô.',
    'std_intensity': 'Độ biến thiên cường độ sáng. Cho biết khối u có đồng nhất hay không.',
    'bbox_width': 'Chiều rộng của hình chữ nhật bao quanh khối u.',
    'bbox_height': 'Chiều cao của hình chữ nhật bao quanh khối u.'
  };
  
  // ===== 🎨 FEATURE IMPORTANCE COLORS =====
  const IMPORTANCE_COLORS = {
    critical: { threshold: 40, color: '#ff5252', label: '🔴 Yếu tố chính', rgb: '255, 82, 82' },
    high: { threshold: 20, color: '#ff9100', label: '🟡 Yếu tố quan trọng', rgb: '255, 145, 0' },
    medium: { threshold: 10, color: '#00e5ff', label: '🟢 Yếu tố phụ', rgb: '0, 229, 255' },
    low: { threshold: 0, color: '#8899b0', label: '⚪ Ảnh hưởng nhỏ', rgb: '136, 153, 176' }
  };
  
  // ===== LOGGING UTILITY =====
  function log(message, data = null) {
    if (!DEBUG) return;
    const style = 'color: #00e5ff; font-weight: bold;';
    if (data) {
      console.log(`%c[XAI] ${message}`, style, data);
    } else {
      console.log(`%c[XAI] ${message}`, style);
    }
  }
  
  function warn(message, data = null) {
    const style = 'color: #ff9100; font-weight: bold;';
    if (data) {
      console.warn(`%c[XAI] ${message}`, style, data);
    } else {
      console.warn(`%c[XAI] ${message}`, style);
    }
  }
  
  function error(message, data = null) {
    const style = 'color: #ff5252; font-weight: bold;';
    if (data) {
      console.error(`%c[XAI] ${message}`, style, data);
    } else {
      console.error(`%c[XAI] ${message}`, style);
    }
  }
  
  // ===== MAIN UI CONTROLLER =====
  window.XAISimilarUI = {
    
    // State management
    state: {
      currentXAIData: null,
      currentSimilarData: null,
      isInitialized: false
    },
    
    // ===== INITIALIZATION =====
    init: function() {
      log('🚀 Initializing XAI & Similar Cases UI (Vietnamese Mode)');
      
      try {
        this.setupEventListeners();
        this.setupDiagnosisIntegration();
        this.state.isInitialized = true;
        log('✅ UI initialized and ready');
      } catch (e) {
        error('Failed to initialize', e);
      }
    },
    
    // ===== EVENT LISTENERS SETUP =====
    setupEventListeners: function() {
      log('Setting up event listeners');
      
      document.addEventListener('diagnosisComplete', (event) => {
        log('📥 Diagnosis complete event received', event.detail);
        
        if (event.detail && event.detail.xaiData) {
          this.state.currentXAIData = event.detail.xaiData;
          window.lastXAIData = event.detail.xaiData;
          
          this.renderXAIDashboard(event.detail.xaiData);
          this.showXAIPanel();
        }
        
        if (event.detail && event.detail.similarData) {
          this.state.currentSimilarData = event.detail.similarData;
          window.lastSimilarData = event.detail.similarData;
          
          this.renderSimilarCases(event.detail.similarData);
        }
      });
      
      document.addEventListener('diagnosisError', (event) => {
        warn('Diagnosis error event received', event.detail);
        this.showXAIError(event.detail?.message || 'Unknown error');
      });
    },
    
    // ===== DIAGNOSIS INTEGRATION =====
    setupDiagnosisIntegration: function() {
      log('Setting up diagnosis integration');
      
      window.XAISimilarUI.renderDiagnosisResults = (diagnosisData) => {
        log('Rendering diagnosis results', diagnosisData);
        
        if (diagnosisData.xai) {
          window.XAISimilarUI.renderXAIDashboard(diagnosisData.xai);
          window.XAISimilarUI.showXAIPanel();
        }
        
        if (diagnosisData.similar_cases) {
          window.XAISimilarUI.renderSimilarCases({
            similar_cases: diagnosisData.similar_cases,
            search_time_ms: diagnosisData.search_time_ms || 0,
            total_cases: diagnosisData.total_cases || 0
          });
          window.XAISimilarUI.showSimilarPanel();
        }
      };
    },
    
    // ===== FETCH SIMILAR CASES =====
    fetchSimilarCases: async function(imageFile) {
      log('🔍 Fetching similar cases...');
      
      if (!imageFile) {
        warn('No image file provided');
        this.showSimilarPlaceholder('no-image');
        return null;
      }
      
      try {
        const formData = new FormData();
        formData.append('file', imageFile);
        formData.append('k', 5);
        
        log('📤 Sending request to /api/similar/find');
        
        const response = await fetch(`${API_BASE}/similar/find`, {
          method: 'POST',
          body: formData
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          log('⚠️ Similar search unavailable:', errorData);
          
          if (response.status === 503) {
            this.showSimilarPlaceholder('index-not-built', errorData);
            return null;
          }
          
          throw new Error(errorData.message || 'Similar search failed');
        }
        
        const data = await response.json();
        log('✅ Similar cases received:', data);
        
        this.state.currentSimilarData = data;
        window.lastSimilarData = data;
        
        this.renderSimilarCases(data);  
        
        return data;
        
      } catch (error) {
        warn('❌ Similar cases fetch error:', error);
        this.showSimilarPlaceholder('error', error);
        return null;
      }
    },
    
    // ===== RENDER XAI DASHBOARD =====
    renderXAIDashboard: function(xaiData) {
      const panel = document.getElementById('xaiPanel');
      
      if (!panel) {
        error('XAI panel not found in DOM');
        return;
      }
      
      log('📊 Rendering XAI Dashboard', xaiData);
      
      if (!xaiData) {
        this.showXAIError('No XAI data provided');
        return;
      }
      
      const hasGradCAM = xaiData.gradcam && Object.keys(xaiData.gradcam).length > 0;
      const hasRuleBased = xaiData.rule_based && Object.keys(xaiData.rule_based).length > 0;
      const hasSHAP = xaiData.shap && Object.keys(xaiData.shap).length > 0;
      const hasInsights = xaiData.combined_insights && xaiData.combined_insights.length > 0;
      
      const hasAnyData = hasGradCAM || hasRuleBased || hasSHAP;
      
      if (!hasAnyData && xaiData.error) {
        this.showXAIError(xaiData.error);
        return;
      }
      
      log('XAI Data Available:', {
        gradCAM: hasGradCAM,
        ruleBased: hasRuleBased,
        shap: hasSHAP,
        insights: hasInsights
      });
      
      const html = `
        <div class="xai-container" style="${this.styles.container}">
          
          <!-- Header -->
          <div class="xai-header" style="${this.styles.header}">
            <h2 style="${this.styles.title}">🔍 Phân Tích AI Có Giải Thích</h2>
            <p style="${this.styles.subtitle}">Phân tích đa phương pháp để hiểu quyết định của AI</p>
          </div>
          
          <!-- Cards Grid -->
          <div class="xai-grid" style="${this.styles.grid}">
            ${hasGradCAM ? this.renderGradCAMCard(xaiData.gradcam) : ''}
            ${hasRuleBased ? this.renderRuleBasedCard(xaiData.rule_based) : ''}
            ${hasSHAP ? this.renderSHAPCard(xaiData.shap) : ''}
          </div>
          
          <!-- Combined Insights -->
          ${hasInsights ? `
            <div class="xai-card insights-card" style="${this.styles.insightsCard}">
              <div style="${this.styles.cardHeader}">
                <h3 style="${this.styles.cardTitle}">💡 Kết Luận Tổng Hợp</h3>
                <span style="${this.styles.badge}">TẤT CẢ PHƯƠNG PHÁP</span>
              </div>
              <ul style="${this.styles.insightsList}">
                ${xaiData.combined_insights.map((insight, idx) => `
                  <li style="${this.styles.insightItem}">
                    <span style="${this.styles.insightEmoji}">${this.getInsightEmoji(insight)}</span>
                    ${this.escapeHtml(insight)}
                  </li>
                `).join('')}
              </ul>
            </div>
          ` : ''}
          
          <!-- Status Footer -->
          <div style="${this.styles.statusFooter}">
            <span style="color: #5a7a99; font-size: 11px;">
              ✅ Phân tích hoàn tất | ${new Date().toLocaleTimeString('vi-VN')}
            </span>
          </div>
        </div>
      `;
      
      panel.innerHTML = html;
      panel.style.display = 'block';
      
      log('✅ XAI dashboard rendered successfully');
    },
    
    // ===== GRAD-CAM CARD =====
    renderGradCAMCard: function(gradcam) {
      if (!gradcam) return '';
      
      const attScore = Math.round((gradcam.attention_score || 0) * 100);
      const technicalInfo = gradcam.technical_info || {};
      const sliceInfo = gradcam.slice_info || {};
      
      let cardHTML = `
        <div class="xai-card" style="${this.styles.card}">
          
          <div style="${this.styles.cardHeader}">
            <h3 style="${this.styles.cardTitle}">🧠 Trực Quan Hóa Grad-CAM</h3>
            <span style="${this.styles.badge}">CNN</span>
          </div>
          
          <!-- Attention Score -->
          <div style="${this.styles.scoreBox}">
            <div style="color: #8899b0; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">
              Mức Độ Tập Trung Của CNN
            </div>
            <div style="color: #00e5ff; font-size: 32px; font-weight: bold; margin-bottom: 8px;">
              ${attScore}%
            </div>
            
            <!-- ✅ DISCLAIMER -->
            <div style="background: rgba(255, 145, 0, 0.1); padding: 10px; border-radius: 4px; margin-bottom: 12px; border-left: 3px solid #ff9100;">
              <div style="color: #ff9100; font-size: 10px; line-height: 1.5;">
                <strong>⚠️ Lưu ý:</strong> Đây là mức độ <strong>tập trung</strong> của CNN vào vùng khối u (attention focus), 
                KHÔNG phải độ tin cậy dự đoán chung. Độ tin cậy dự đoán hiển thị ở phần "Báo cáo chẩn đoán".
              </div>
            </div>
            
            <div style="${this.styles.progressBar}">
              <div style="height: 100%; width: ${attScore}%; ${this.styles.progressFill}"></div>
            </div>
          </div>
          
          <!-- Technical Details -->
          <div style="${this.styles.infoBox}; margin-bottom: 12px;">
            <h4 style="color: #8899b0; margin: 0 0 8px 0; font-size: 11px; text-transform: uppercase;">
              ⚙️ Chi Tiết Kỹ Thuật
            </h4>
            <ul style="margin: 0; padding-left: 16px; list-style: none; font-size: 10px; color: #c1cfe8;">
              <li style="margin-bottom: 4px;">
                <strong>Lớp mạng:</strong> ${technicalInfo.layer_name || 'Conv2D cuối'}
              </li>
              <li style="margin-bottom: 4px;">
                <strong>Vị trí:</strong> ${technicalInfo.position || 'Encoder bottleneck'}
              </li>
              <li style="margin-bottom: 4px;">
                <strong>Phương pháp:</strong> ${technicalInfo.gradient_method || 'Grad-CAM'}
              </li>
              <li style="margin-bottom: 4px;">
                <strong>Lát cắt:</strong> ${sliceInfo.type || 'axial'} - ${sliceInfo.resolution || '256x256'}
              </li>
              <li>
                <strong>Tổng hợp:</strong> ${technicalInfo.aggregation_method || 'Không gian 2D'}
              </li>
            </ul>
          </div>
          
          <!-- Confidence Level -->
          ${gradcam.confidence_level ? `
            <div style="padding: 8px; background: rgba(${this.getConfidenceColor(gradcam.confidence_level)}, 0.1); 
              border-left: 3px solid ${this.getConfidenceColorHex(gradcam.confidence_level)}; 
              border-radius: 4px; margin-bottom: 12px;">
              <div style="color: #8899b0; font-size: 9px; text-transform: uppercase; margin-bottom: 4px;">
                Mức Độ Tin Cậy CNN
              </div>
              <div style="color: ${this.getConfidenceColorHex(gradcam.confidence_level)}; 
                font-size: 14px; font-weight: bold;">
                ${this.translateConfidenceLevel(gradcam.confidence_level)}
              </div>
            </div>
          ` : ''}
          
          <!-- Interpretation -->
          ${gradcam.interpretation ? `
            <div style="padding: 10px; background: rgba(136, 153, 176, 0.05); 
              border-radius: 4px; margin-bottom: 12px;">
              <div style="color: #c1cfe8; font-size: 10px; line-height: 1.5;">
                💡 ${this.escapeHtml(gradcam.interpretation)}
              </div>
            </div>
          ` : ''}
      `;
      
      // Overlay image
      if (gradcam.overlay_base64) {
        cardHTML += `
          <div style="margin: 12px 0;">
            <img src="${gradcam.overlay_base64}" alt="Grad-CAM Overlay" 
              style="${this.styles.image}"/>
            <p style="color: #5a7a99; font-size: 11px; text-align: center; margin: 6px 0 0 0;">
              Bản đồ nhiệt tập trung chồng lên ảnh gốc
            </p>
          </div>
        `;
      }
      
      // Heatmap image
      if (gradcam.heatmap_base64) {
        cardHTML += `
          <div style="margin: 12px 0;">
            <img src="${gradcam.heatmap_base64}" alt="Grad-CAM Heatmap" 
              style="${this.styles.image}"/>
            <p style="color: #5a7a99; font-size: 11px; text-align: center; margin: 6px 0 0 0;">
              Bản đồ tập trung thuần
            </p>
          </div>
        `;
      }
      
      // Focused regions
      if (gradcam.focused_regions && gradcam.focused_regions.length > 0) {
        cardHTML += `
          <div style="${this.styles.infoBox}">
            <h4 style="color: #8899b0; margin: 0 0 8px 0; font-size: 12px; text-transform: uppercase;">
              Vùng Tập Trung
            </h4>
            <ul style="margin: 0; padding-left: 16px; list-style: none;">
              ${gradcam.focused_regions.slice(0, 3).map((region, i) => `
                <li style="color: #c1cfe8; font-size: 12px; margin-bottom: 4px;">
                  Vùng ${i+1}: <span style="color: #00e5ff; font-weight: bold;">
                    ${Math.round((region.attention || 0) * 100)}%
                  </span> tập trung
                </li>
              `).join('')}
            </ul>
          </div>
        `;
      }
      
      cardHTML += `</div>`;
      return cardHTML;
    },
    
    // ===== RULE-BASED CARD =====
    renderRuleBasedCard: function(rules) {
      if (!rules) return '';
      
      const riskLevel = rules.risk_level || 'Unknown';
      const riskColors = {
        'High': { bg: '#ff5252', rgb: '255, 82, 82', vi: 'Cao' },
        'Medium': { bg: '#ff9100', rgb: '255, 145, 0', vi: 'Trung Bình' },
        'Low': { bg: '#00c853', rgb: '0, 200, 83', vi: 'Thấp' }
      };
      const riskColor = riskColors[riskLevel] || { bg: '#8899b0', rgb: '136, 153, 176', vi: 'Không xác định' };
      
      let cardHTML = `
        <div class="xai-card" style="${this.styles.card}">
          <div style="${this.styles.cardHeader}">
            <h3 style="${this.styles.cardTitle}">📊 Phân Tích Thống Kê</h3>
            <span style="${this.styles.badge}">QUY TẮC</span>
          </div>
          
          <!-- Risk Level -->
          <div style="padding: 12px; border-left: 3px solid ${riskColor.bg}; 
            background: rgba(${riskColor.rgb}, 0.1); border-radius: 4px; margin-bottom: 16px;">
            <div style="color: #8899b0; font-size: 11px; text-transform: uppercase; 
              letter-spacing: 0.5px; margin-bottom: 6px;">Mức Độ Rủi Ro</div>
            <div style="color: ${riskColor.bg}; font-size: 28px; font-weight: bold;">
              ${riskColor.vi}
            </div>
          </div>
          
          <!-- Measurements Grid -->
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 16px;">
            <div style="${this.styles.infoBox}">
              <div style="color: #5a7a99; font-size: 10px; text-transform: uppercase;">Diện Tích Khối U</div>
              <div style="color: #00e5ff; font-size: 18px; font-weight: bold; margin-top: 4px;">
                ${rules.tumor_area_mm2 !== undefined ? rules.tumor_area_mm2.toFixed(1) : 'N/A'}
              </div>
              <div style="color: #5a7a99; font-size: 9px;">mm²</div>
            </div>
            <div style="${this.styles.infoBox}">
              <div style="color: #5a7a99; font-size: 10px; text-transform: uppercase;">Phủ Não</div>
              <div style="color: #00e5ff; font-size: 18px; font-weight: bold; margin-top: 4px;">
                ${rules.tumor_ratio !== undefined ? rules.tumor_ratio.toFixed(1) : 'N/A'}
              </div>
              <div style="color: #5a7a99; font-size: 9px;">%</div>
            </div>
            <div style="${this.styles.infoBox}">
              <div style="color: #5a7a99; font-size: 10px; text-transform: uppercase;">Vị Trí</div>
              <div style="color: #00e5ff; font-size: 14px; font-weight: bold; margin-top: 4px;">
                ${rules.location || 'Không xác định'}
              </div>
            </div>
            <div style="${this.styles.infoBox}">
              <div style="color: #5a7a99; font-size: 10px; text-transform: uppercase;">Mức Độ</div>
              <div style="color: #00e5ff; font-size: 14px; font-weight: bold; margin-top: 4px;">
                ${rules.severity || 'Trung bình'}
              </div>
            </div>
          </div>
      `;
      
      // Rules triggered
      if (rules.rules_triggered && rules.rules_triggered.length > 0) {
        cardHTML += `
          <div style="${this.styles.infoBox}; margin-bottom: 12px;">
            <h4 style="color: #8899b0; margin: 0 0 8px 0; font-size: 12px; text-transform: uppercase;">
              ✓ Quy Tắc Đã Kích Hoạt
            </h4>
            <ul style="margin: 0; padding-left: 16px; list-style: none;">
              ${rules.rules_triggered.slice(0, 3).map(rule => `
                <li style="color: #c1cfe8; font-size: 12px; margin-bottom: 4px;">
                  ✓ ${this.escapeHtml(rule)}
                </li>
              `).join('')}
            </ul>
          </div>
        `;
      }
      
      // Warnings
      if (rules.warnings && rules.warnings.length > 0) {
        cardHTML += `
          <div style="padding: 12px; background: rgba(255, 82, 82, 0.1); 
            border-left: 3px solid #ff5252; border-radius: 4px;">
            <h4 style="color: #ff5252; margin: 0 0 8px 0; font-size: 12px; text-transform: uppercase;">
              ⚠️ Cảnh Báo Lâm Sàng
            </h4>
            <ul style="margin: 0; padding-left: 16px; list-style: none;">
              ${rules.warnings.slice(0, 3).map(warning => `
                <li style="color: #ffb3b3; font-size: 12px; margin-bottom: 4px;">
                  ⚠️ ${this.escapeHtml(warning)}
                </li>
              `).join('')}
            </ul>
          </div>
        `;
      }
      
      cardHTML += `</div>`;
      return cardHTML;
    },
    
    // ===== ✅ SHAP CARD (VIETNAMESE + FIXED) =====
    renderSHAPCard: function(shap) {
      if (!shap) return '';
      
      const topFeatures = shap.top_features || [];
      const featureImportance = shap.feature_importance || {};
      
      log('Rendering SHAP card (Vietnamese)', { topFeatures, featureImportance });
      
      let cardHTML = `
        <div class="xai-card" style="${this.styles.card}">
          <div style="${this.styles.cardHeader}">
            <h3 style="${this.styles.cardTitle}">📈 Tầm Quan Trọng Của Các Tính Năng</h3>
            <span style="${this.styles.badge}">SHAP</span>
          </div>
      `;
      
      if (topFeatures.length > 0) {
        cardHTML += `
          <div>
            <h4 style="color: #8899b0; margin: 0 0 12px 0; font-size: 12px; text-transform: uppercase;">
              Các Tính Năng Đóng Góp Hàng Đầu (Tầm Quan Trọng Tương Đối)
            </h4>
            
            <!-- ✅ EXPLANATION BOX -->
            <div style="background: rgba(136, 153, 176, 0.05); padding: 12px; border-radius: 4px; margin-bottom: 16px; border-left: 3px solid #00e5ff;">
              <div style="color: #c1cfe8; font-size: 10px; line-height: 1.6;">
                💡 <strong>Giải thích:</strong> % đóng góp <strong>tương đối</strong> của mỗi tính năng vào dự đoán cuối cùng. 
                Tổng tất cả tính năng = 100%. Con số càng cao = ảnh hưởng càng lớn đến kết quả.
              </div>
            </div>
            
            <div style="display: flex; flex-direction: column; gap: 12px;">
              ${topFeatures.slice(0, 5).map((feature, index) => {
                const importance = featureImportance[feature] || 0;
                const importancePercent = Math.round(importance * 100);
                
                // Get Vietnamese name and description
                const featureNameVI = FEATURE_NAMES_VI[feature] || feature;
                const featureDesc = FEATURE_DESCRIPTIONS_VI[feature] || '';
                
                // Determine importance level and color
                const importanceLevel = this.getImportanceLevel(importancePercent);
                
                log(`SHAP Feature ${index+1}: ${feature} (${featureNameVI}) = ${importance} → ${importancePercent}%`);
                
                return `
                  <div style="padding: 12px; background: rgba(${importanceLevel.rgb}, 0.08); 
                    border-radius: 6px; border-left: 3px solid ${importanceLevel.color};">
                    
                    <!-- Feature Header -->
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                      <div style="flex: 1;">
                        <div style="color: #c1cfe8; font-size: 13px; font-weight: 600; margin-bottom: 2px;">
                          ${this.escapeHtml(featureNameVI)}
                        </div>
                        <div style="color: #5a7a99; font-size: 9px;">
                          ${this.escapeHtml(feature)}
                        </div>
                      </div>
                      <div style="text-align: right; margin-left: 12px;">
                        <div style="color: ${importanceLevel.color}; font-size: 18px; font-weight: bold;">
                          ${importancePercent}%
                        </div>
                        <div style="color: #5a7a99; font-size: 8px; text-transform: uppercase;">
                          Đóng góp
                        </div>
                      </div>
                    </div>
                    
                    <!-- Progress Bar -->
                    <div style="${this.styles.progressBar}; margin-bottom: 8px;">
                      <div style="height: 100%; width: ${importancePercent}%; background: ${importanceLevel.color}; 
                        border-radius: 2px; transition: width 0.3s ease;"></div>
                    </div>
                    
                    <!-- Importance Label -->
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: ${featureDesc ? '8px' : '0'};">
                      <span style="color: ${importanceLevel.color}; font-size: 10px; font-weight: 600;">
                        ${importanceLevel.label}
                      </span>
                      <span style="color: #5a7a99; font-size: 9px;">
                        ${this.getImportanceExplanation(importancePercent)}
                      </span>
                    </div>
                    
                    <!-- Feature Description (if available) -->
                    ${featureDesc ? `
                      <div style="background: rgba(10, 14, 26, 0.3); padding: 8px; border-radius: 4px; margin-top: 8px;">
                        <div style="color: #8899b0; font-size: 9px; line-height: 1.5;">
                          ℹ️ ${this.escapeHtml(featureDesc)}
                        </div>
                      </div>
                    ` : ''}
                  </div>
                `;
              }).join('')}
            </div>
          </div>
        `;
      } else {
        cardHTML += `
          <p style="color: #5a7a99; font-size: 12px; text-align: center; padding: 20px;">
            Không có dữ liệu tầm quan trọng tính năng
          </p>
        `;
      }
      
      cardHTML += `</div>`;
      return cardHTML;
    },
    
    // ===== RENDER SIMILAR CASES =====
    renderSimilarCases: function(similarData) {
      const panel = document.getElementById('similarPanel');
      
      if (!panel) {
        warn('Similar panel not found in DOM');
        return;
      }
      
      log('📊 Rendering Similar Cases', similarData);
      
      if (!similarData || !similarData.similar_cases || similarData.similar_cases.length === 0) {
        this.showSimilarPlaceholder('no-results');
        return;
      }
      
      const html = `
        <div style="padding: 30px; background: #0a0e1a; border-radius: 12px; min-height: 100vh;">
          
          <!-- Header -->
          <div style="margin-bottom: 30px;">
            <h2 style="color: #00e5ff; margin: 0 0 10px 0; font-size: 24px; font-weight: bold;">
              🔎 Các Ca Bệnh Tương Tự
            </h2>
            <p style="color: #5a7a99; margin: 0; font-size: 13px;">
              Tìm thấy ${similarData.similar_cases.length} ca bệnh tương tự trong ${similarData.search_time_ms.toFixed(1)}ms
              ${similarData.total_cases ? ` (đã tìm ${similarData.total_cases} ca bệnh)` : ''}
            </p>
          </div>
          
          <!-- Grid -->
          <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 20px;">
            ${similarData.similar_cases.map(caseItem => this.renderCaseCard(caseItem)).join('')}
          </div>
          
          <!-- Footer Info -->
          <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #1e3a52; 
            text-align: center; color: #5a7a99; font-size: 11px;">
            ✅ Tìm kiếm hoàn tất | ${new Date().toLocaleTimeString('vi-VN')}
          </div>
        </div>
      `;
      
      panel.innerHTML = html;
      panel.style.display = 'block';
      
      log('✅ Similar cases rendered successfully');
    },
    
    // ===== RENDER CASE CARD =====
    renderCaseCard: function(caseItem) {
      const similarity = Math.round((caseItem.similarity_score || 0) * 100);
      const statusColor = caseItem.has_tumor ? '#ff5252' : '#00c853';
      const statusText = caseItem.has_tumor ? '🔴 Phát hiện khối u' : '🟢 Không có khối u';
      
      return `
        <div style="padding: 20px; border: 1px solid #1e3a52; border-radius: 8px; 
          background: linear-gradient(135deg, #0f1f2e 0%, #1a2f42 100%); position: relative;">
          
          <!-- Rank Badge -->
          <div style="position: absolute; top: 16px; right: 16px; background: #0a0e1a; 
            color: #00e5ff; padding: 6px 12px; border-radius: 4px; font-size: 12px; 
            font-weight: bold; box-shadow: 0 2px 8px rgba(0,229,255,0.3);">
            #${caseItem.rank || '?'}
          </div>
          
          <!-- Thumbnail -->
          <div style="width: 100%; height: 180px; background: #0a1a28; border-radius: 6px; 
            display: flex; align-items: center; justify-content: center; margin-bottom: 16px; 
            overflow: hidden; border: 1px solid #1e3a52;">
            ${this.renderThumbnail(caseItem)}
          </div>
          
          <!-- Similarity Score -->
          <div style="padding: 12px; background: rgba(0, 229, 255, 0.1); 
            border-left: 3px solid #00e5ff; border-radius: 4px; margin-bottom: 12px;">
            <div style="color: #5a7a99; font-size: 10px; text-transform: uppercase; 
              letter-spacing: 0.5px; margin-bottom: 6px;">
              Độ Tương Đồng
            </div>
            <div style="color: #00e5ff; font-size: 24px; font-weight: bold; margin-bottom: 8px;">
              ${similarity}%
            </div>
            <div style="width: 100%; height: 4px; background: #0a1a28; border-radius: 2px; overflow: hidden;">
              <div style="height: 100%; width: ${similarity}%; 
                background: linear-gradient(90deg, #00e5ff, #00a3cc); border-radius: 2px;">
              </div>
            </div>
          </div>
          
          <!-- Status -->
          <div style="padding: 10px; background: rgba(${this.hexToRgb(statusColor)}, 0.1); 
            border-left: 3px solid ${statusColor}; border-radius: 4px; margin-bottom: 12px;">
            <div style="color: ${statusColor}; font-size: 13px; font-weight: bold;">
              ${statusText}
            </div>
          </div>
          
          <!-- Metadata Grid -->
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 11px; color: #5a7a99;">
            <div style="padding: 8px; background: rgba(82, 143, 204, 0.05); border-radius: 4px;">
              <div style="color: #8899b0; font-size: 9px; text-transform: uppercase; margin-bottom: 4px;">
                Mã Ca Bệnh
              </div>
              <div style="color: #c1cfe8; font-weight: 500;">
                ${caseItem.case_id !== undefined ? caseItem.case_id : 'N/A'}
              </div>
            </div>
            <div style="padding: 8px; background: rgba(82, 143, 204, 0.05); border-radius: 4px;">
              <div style="color: #8899b0; font-size: 9px; text-transform: uppercase; margin-bottom: 4px;">
                Khoảng Cách
              </div>
              <div style="color: #c1cfe8; font-weight: 500;">
                ${(caseItem.distance || 0).toFixed(3)}
              </div>
            </div>
            <div style="padding: 8px; background: rgba(82, 143, 204, 0.05); border-radius: 4px;">
              <div style="color: #8899b0; font-size: 9px; text-transform: uppercase; margin-bottom: 4px;">
                Nguồn
              </div>
              <div style="color: #c1cfe8; font-weight: 500;">
                ${caseItem.source || 'Không rõ'}
              </div>
            </div>
            <div style="padding: 8px; background: rgba(82, 143, 204, 0.05); border-radius: 4px;">
              <div style="color: #8899b0; font-size: 9px; text-transform: uppercase; margin-bottom: 4px;">
                Bệnh Nhân
              </div>
              <div style="color: #c1cfe8; font-weight: 500;">
                ${caseItem.patient_id || 'N/A'}
              </div>
            </div>
          </div>
          
          <!-- Filename -->
          ${caseItem.filename ? `
            <div style="margin-top: 12px; padding: 8px; background: #0a1a28; border-radius: 4px;">
              <div style="color: #5a7a99; font-size: 9px; text-transform: uppercase; margin-bottom: 4px;">
                Tên File
              </div>
              <div style="color: #8899b0; font-size: 10px; word-break: break-all;">
                ${this.escapeHtml(caseItem.filename)}
              </div>
            </div>
          ` : ''}
        </div>
      `;
    },
    
    // ===== RENDER THUMBNAIL =====
    renderThumbnail: function(caseItem) {
      if (caseItem.thumbnail) {
        return `<img src="${caseItem.thumbnail}" alt="Ca bệnh ${caseItem.case_id}" 
          style="width: 100%; height: 100%; object-fit: cover;"/>`;
      }
      
      if (caseItem.filename) {
        const imgPath = `/data/images/${caseItem.filename}`;
        return `<img src="${imgPath}" alt="${caseItem.filename}" 
          style="width: 100%; height: 100%; object-fit: cover;" 
          onerror="this.parentElement.innerHTML='<div style=\\'color: #5a7a99; font-size: 12px;\\'>Không có hình ảnh</div>'"/>`;
      }
      
      return `<div style="color: #5a7a99; font-size: 12px;">📄 Không có hình ảnh</div>`;
    },
    
    // ===== SHOW PLACEHOLDER =====
    showSimilarPlaceholder: function(reason, data = null) {
      const panel = document.getElementById('similarPanel');
      if (!panel) return;
      
      let html = '';
      
      switch(reason) {
        case 'index-not-built':
          html = `
            <div style="padding: 80px 40px; text-align: center; max-width: 600px; margin: 0 auto;">
              <div style="font-size: 64px; margin-bottom: 24px;">🔧</div>
              <h2 style="color: #ff9100; font-size: 24px; margin: 0 0 16px 0;">
                Tính Năng Tìm Ca Bệnh Tương Tự Chưa Khả Dụng
              </h2>
              <p style="color: #8899b0; font-size: 14px; line-height: 1.6; margin: 0 0 24px 0;">
                Chỉ mục tương đồng FAISS chưa được xây dựng.
              </p>
              <div style="background: #0a1a28; padding: 16px; border-radius: 8px; border: 1px solid #1e3a52; margin-bottom: 16px;">
                <div style="color: #5a7a99; font-size: 12px; margin-bottom: 8px;">
                  Để kích hoạt tính năng này, chạy:
                </div>
                <code style="display: block; background: #0f1f2e; color: #00e5ff; padding: 12px; 
                  border-radius: 4px; font-family: 'Courier New', monospace; font-size: 13px;">
                  python dataset_scripts/build_faiss_index.py
                </code>
              </div>
              ${data && data.details ? `
                <p style="color: #5a7a99; font-size: 11px; margin: 16px 0 0 0;">
                  ${this.escapeHtml(data.details)}
                </p>
              ` : ''}
            </div>
          `;
          break;
          
        case 'no-results':
          html = `
            <div style="padding: 80px 40px; text-align: center;">
              <div style="font-size: 48px; margin-bottom: 16px;">🔍</div>
              <h3 style="color: #5a7a99; margin: 0 0 8px 0;">Không Tìm Thấy Ca Bệnh Tương Tự</h3>
              <p style="color: #5a7a99; font-size: 12px;">
                Thử tải lên một ảnh MRI khác.
              </p>
            </div>
          `;
          break;
          
        case 'no-image':
          html = `
            <div style="padding: 80px 40px; text-align: center;">
              <div style="font-size: 48px; margin-bottom: 16px;">📤</div>
              <h3 style="color: #5a7a99; margin: 0 0 8px 0;">Tải Lên Ảnh MRI</h3>
              <p style="color: #5a7a99; font-size: 12px;">
                Tải lên và chẩn đoán ảnh MRI để tìm ca bệnh tương tự.
              </p>
            </div>
          `;
          break;
          
        case 'error':
        default:
          html = `
            <div style="padding: 80px 40px; text-align: center;">
              <div style="font-size: 48px; margin-bottom: 16px; color: #ff5252;">⚠️</div>
              <h3 style="color: #ff5252; margin: 0 0 8px 0;">Lỗi Khi Tải Ca Bệnh Tương Tự</h3>
              <p style="color: #8899b0; font-size: 12px;">
                ${data ? this.escapeHtml(String(data.message || data)) : 'Lỗi không xác định'}
              </p>
            </div>
          `;
      }
      
      panel.innerHTML = `<div style="background: #0a0e1a; border-radius: 12px; min-height: 100vh;">${html}</div>`;
      panel.style.display = 'block';
    },
    
    // ===== SHOW XAI ERROR =====
    showXAIError: function(errorMessage) {
      const panel = document.getElementById('xaiPanel');
      if (!panel) return;
      
      error('XAI Error', errorMessage);
      
      panel.innerHTML = `
        <div style="${this.styles.container}">
          <div style="padding: 60px 40px; text-align: center; color: #ff5252;">
            <div style="font-size: 48px; margin-bottom: 24px;">⚠️</div>
            <h2 style="color: #ff5252; margin: 0 0 16px 0; font-size: 18px;">
              Phân Tích XAI Không Khả Dụng
            </h2>
            <p style="color: #8899b0; font-size: 12px; margin: 0;">
              ${this.escapeHtml(errorMessage)}
            </p>
          </div>
        </div>
      `;
      
      panel.style.display = 'block';
    },
    
    // ===== SHOW/HIDE PANELS =====
    showXAIPanel: function() {
      const panel = document.getElementById('xaiPanel');
      if (panel) panel.style.display = 'block';
    },
    
    showSimilarPanel: function() {
      const panel = document.getElementById('similarPanel');
      if (panel) panel.style.display = 'block';
    },
    
    hideXAIPanel: function() {
      const panel = document.getElementById('xaiPanel');
      if (panel) panel.style.display = 'none';
    },
    
    hideSimilarPanel: function() {
      const panel = document.getElementById('similarPanel');
      if (panel) panel.style.display = 'none';
    },
    
    // ===== UTILITIES =====
    
    getImportanceLevel: function(percent) {
      if (percent > 40) return IMPORTANCE_COLORS.critical;
      if (percent > 20) return IMPORTANCE_COLORS.high;
      if (percent > 10) return IMPORTANCE_COLORS.medium;
      return IMPORTANCE_COLORS.low;
    },
    
    getImportanceExplanation: function(percent) {
      if (percent > 40) return `Ảnh hưởng rất lớn (>${percent}%)`;
      if (percent > 20) return `Ảnh hưởng đáng kể (${percent}%)`;
      if (percent > 10) return `Ảnh hưởng vừa phải (${percent}%)`;
      return `Ảnh hưởng nhỏ (${percent}%)`;
    },
    
    translateConfidenceLevel: function(level) {
      const translations = {
        'HIGH': 'Cao',
        'MEDIUM': 'Trung Bình',
        'LOW': 'Thấp'
      };
      return translations[level] || level;
    },
    
    getConfidenceColor: function(level) {
      const colors = {
        'HIGH': '0, 200, 83',
        'MEDIUM': '255, 145, 0',
        'LOW': '255, 82, 82'
      };
      return colors[level] || '136, 153, 176';
    },
    
    getConfidenceColorHex: function(level) {
      const colors = {
        'HIGH': '#00c853',
        'MEDIUM': '#ff9100',
        'LOW': '#ff5252'
      };
      return colors[level] || '#8899b0';
    },
    
    hexToRgb: function(hex) {
      const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      if (!result) return '255, 0, 0';
      return [
        parseInt(result[1], 16),
        parseInt(result[2], 16),
        parseInt(result[3], 16)
      ].join(', ');
    },
    
    getInsightEmoji: function(insight) {
      const text = String(insight).toLowerCase();
      if (text.includes('high') || text.includes('risk') || text.includes('cao')) return '⚠️';
      if (text.includes('low') || text.includes('normal') || text.includes('thấp')) return '✅';
      if (text.includes('location') || text.includes('vị trí')) return '📍';
      if (text.includes('size') || text.includes('kích thước')) return '📏';
      if (text.includes('cnn')) return '🧠';
      if (text.includes('tumor') || text.includes('khối u')) return '⚕️';
      return '💡';
    },
    
    escapeHtml: function(text) {
      const div = document.createElement('div');
      div.textContent = String(text || '');
      return div.innerHTML;
    },
    
    // ===== STYLES OBJECT =====
    styles: {
      container: 'padding: 30px; background: #0a0e1a; border-radius: 12px;',
      header: 'margin-bottom: 30px;',
      title: 'color: #00e5ff; margin: 0 0 10px 0; font-size: 24px; font-weight: bold;',
      subtitle: 'color: #5a7a99; margin: 0; font-size: 13px;',
      grid: 'display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; margin-bottom: 20px;',
      card: 'padding: 20px; border: 1px solid #1e3a52; border-radius: 8px; background: linear-gradient(135deg, #0f1f2e 0%, #1a2f42 100%);',
      cardHeader: 'display: flex; align-items: center; gap: 10px; margin-bottom: 16px;',
      cardTitle: 'color: #00e5ff; margin: 0; font-size: 16px; font-weight: bold;',
      badge: 'background: #00e5ff; color: #0a0e1a; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;',
      infoBox: 'padding: 10px; background: rgba(82, 143, 204, 0.1); border-radius: 4px;',
      scoreBox: 'padding: 12px; background: rgba(0, 229, 255, 0.1); border-left: 3px solid #00e5ff; border-radius: 4px; margin-bottom: 16px;',
      progressBar: 'width: 100%; height: 4px; background: #0a1a28; border-radius: 2px; overflow: hidden;',
      progressFill: 'height: 100%; background: linear-gradient(90deg, #00e5ff, #00a3cc); border-radius: 2px;',
      image: 'width: 100%; border-radius: 6px; border: 1px solid #1e3a52;',
      insightsCard: 'margin-top: 20px; padding: 20px; border: 1px solid #1e3a52; border-radius: 8px; background: linear-gradient(135deg, #0f1f2e 0%, #1a2f42 100%);',
      insightsList: 'margin: 0; padding-left: 0; list-style: none;',
      insightItem: 'color: #c1cfe8; margin-bottom: 8px; padding-left: 20px; position: relative; font-size: 13px; line-height: 1.5;',
      insightEmoji: 'position: absolute; left: 0; color: #00e5ff; font-weight: bold; width: 16px;',
      statusFooter: 'margin-top: 16px; padding-top: 16px; border-top: 1px solid #1e3a52; text-align: center;'
    }
  };
  
  // ===== AUTO INITIALIZATION =====
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      window.XAISimilarUI.init();
    });
  } else {
    window.XAISimilarUI.init();
  }
  
  log('🎉 XAI & Similar Cases UI module loaded (Vietnamese Mode)');
  
})();

// ===== EXPOSE FOR DEBUGGING =====
window.XAISimilarUIDebug = {
  getState: function() {
    return window.XAISimilarUI.state;
  },
  getLastXAIData: function() {
    return window.lastXAIData;
  },
  getLastSimilarData: function() {
    return window.lastSimilarData;
  },
  render: function(type, data) {
    if (type === 'xai') {
      window.XAISimilarUI.renderXAIDashboard(data);
      window.XAISimilarUI.showXAIPanel();
    } else if (type === 'similar') {
      window.XAISimilarUI.renderSimilarCases(data);
      window.XAISimilarUI.showSimilarPanel();
    }
  }
};

console.log('%c[XAI] Debug mode available: window.XAISimilarUIDebug', 'color: #00e5ff; font-weight: bold;');