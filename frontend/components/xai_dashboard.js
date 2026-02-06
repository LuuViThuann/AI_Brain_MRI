
(function XAIDashboard() {
  
    const API_BASE = 'http://localhost:8000/api';
    
    window.XAIDashboard = {
      
      // Render full xAI dashboard
      renderDashboard: function(xaiData) {
        const container = document.getElementById('xaiPanel');
        
        if (!container) {
          console.error('xAI panel not found');
          return;
        }
        
        container.innerHTML = `
          <div class="xai-grid">
            <!-- Grad-CAM Card -->
            <div class="xai-card gradcam-card">
              <div class="xai-card-header">
                <h3>🔍 Grad-CAM Attention</h3>
                <span class="badge">CNN Visualization</span>
              </div>
              <div class="xai-card-body">
                <div id="gradcamContent"></div>
              </div>
            </div>
            
            <!-- Rule-Based Card -->
            <div class="xai-card rules-card">
              <div class="xai-card-header">
                <h3>📊 Statistical Analysis</h3>
                <span class="badge">Rule-Based</span>
              </div>
              <div class="xai-card-body">
                <div id="rulesContent"></div>
              </div>
            </div>
            
            <!-- SHAP Card -->
            <div class="xai-card shap-card">
              <div class="xai-card-header">
                <h3>📈 Feature Importance</h3>
                <span class="badge">SHAP</span>
              </div>
              <div class="xai-card-body">
                <div id="shapContent"></div>
              </div>
            </div>
            
            <!-- Combined Insights Card -->
            <div class="xai-card insights-card">
              <div class="xai-card-header">
                <h3>💡 Combined Insights</h3>
                <span class="badge">All Methods</span>
              </div>
              <div class="xai-card-body">
                <div id="insightsContent"></div>
              </div>
            </div>
          </div>
        `;
        
        // Render each section
        if (xaiData.gradcam) {
          this.renderGradCAM(xaiData.gradcam);
        }
        
        if (xaiData.rule_based) {
          this.renderRuleBased(xaiData.rule_based);
        }
        
        if (xaiData.shap) {
          this.renderSHAP(xaiData.shap);
        }
        
        if (xaiData.combined_insights) {
          this.renderInsights(xaiData.combined_insights);
        }
      },
      
      // Render Grad-CAM section
      renderGradCAM: function(gradcamData) {
        const container = document.getElementById('gradcamContent');
        
        container.innerHTML = `
          <div class="gradcam-display">
            <div class="attention-score">
              <span class="label">Attention Score:</span>
              <span class="value">${(gradcamData.attention_score * 100).toFixed(1)}%</span>
              <div class="score-bar">
                <div class="score-fill" style="width: ${gradcamData.attention_score * 100}%"></div>
              </div>
            </div>
            
            <div class="heatmap-container">
              <img src="${gradcamData.overlay_base64}" alt="Grad-CAM Overlay" class="heatmap-img"/>
              <p class="image-caption">CNN Attention Heatmap</p>
            </div>
            
            <div class="focused-regions">
              <h4>Focused Regions:</h4>
              <ul>
                ${gradcamData.focused_regions.slice(0, 3).map((region, i) => `
                  <li>Region ${i+1}: Attention ${(region.attention * 100).toFixed(1)}%</li>
                `).join('')}
              </ul>
            </div>
          </div>
        `;
      },
      
      // Render Rule-Based section
      renderRuleBased: function(rulesData) {
        const container = document.getElementById('rulesContent');
        
        const riskColor = {
          'Low': '#00c853',
          'Medium': '#ff9100',
          'High': '#ff5252'
        }[rulesData.risk_level] || '#8899b0';
        
        container.innerHTML = `
          <div class="rules-display">
            <div class="risk-indicator" style="border-color: ${riskColor}">
              <span class="risk-label">Risk Level:</span>
              <span class="risk-value" style="color: ${riskColor}">${rulesData.risk_level}</span>
            </div>
            
            <div class="measurements">
              <div class="measurement-item">
                <span class="label">Tumor Area:</span>
                <span class="value">${rulesData.tumor_area_mm2} mm²</span>
              </div>
              <div class="measurement-item">
                <span class="label">Brain Coverage:</span>
                <span class="value">${rulesData.tumor_ratio}%</span>
              </div>
              <div class="measurement-item">
                <span class="label">Location:</span>
                <span class="value">${rulesData.location}</span>
              </div>
            </div>
            
            <div class="rules-triggered">
              <h4>Rules Triggered:</h4>
              <ul>
                ${rulesData.rules_triggered.map(rule => `<li>${rule}</li>`).join('')}
              </ul>
            </div>
            
            ${rulesData.warnings.length > 0 ? `
              <div class="warnings">
                <h4>⚠ Warnings:</h4>
                <ul>
                  ${rulesData.warnings.map(warning => `<li>${warning}</li>`).join('')}
                </ul>
              </div>
            ` : ''}
          </div>
        `;
      },
      
      // Render SHAP section
      renderSHAP: function(shapData) {
        const container = document.getElementById('shapContent');
        
        container.innerHTML = `
          <div class="shap-display">
            <div class="top-features">
              <h4>Top Contributing Features:</h4>
              <div class="feature-list">
                ${shapData.top_features.slice(0, 5).map((feature, i) => {
                  const importance = shapData.feature_importance[feature];
                  const value = shapData.shap_values[feature];
                  const direction = value > 0 ? '↑ Increases' : '↓ Decreases';
                  const color = value > 0 ? '#00e5ff' : '#ff5252';
                  
                  return `
                    <div class="feature-item">
                      <div class="feature-header">
                        <span class="feature-name">${feature}</span>
                        <span class="feature-importance">${importance.toFixed(3)}</span>
                      </div>
                      <div class="feature-bar">
                        <div class="bar-fill" style="width: ${importance * 100}%; background: ${color}"></div>
                      </div>
                      <div class="feature-direction" style="color: ${color}">${direction} risk</div>
                    </div>
                  `;
                }).join('')}
              </div>
            </div>
            
            ${shapData.contribution_plot ? `
              <div class="contribution-plot">
                <img src="${shapData.contribution_plot}" alt="SHAP Contributions" class="plot-img"/>
              </div>
            ` : ''}
          </div>
        `;
      },
      
      // Render Combined Insights
      renderInsights: function(insights) {
        const container = document.getElementById('insightsContent');
        
        container.innerHTML = `
          <div class="insights-display">
            <ul class="insights-list">
              ${insights.map(insight => `
                <li class="insight-item">
                  <span class="insight-icon">${this.getInsightIcon(insight)}</span>
                  <span class="insight-text">${insight}</span>
                </li>
              `).join('')}
            </ul>
          </div>
        `;
      },
      
      // Get icon for insight
      getInsightIcon: function(insight) {
        if (insight.includes('High') || insight.includes('⚠')) return '⚠';
        if (insight.includes('Low') || insight.includes('✓')) return '✓';
        if (insight.includes('📍')) return '📍';
        if (insight.includes('🔍')) return '🔍';
        if (insight.includes('📊')) return '📊';
        return '•';
      }
      
    };
    
  })();