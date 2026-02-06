(function SimilarCases() {
  
    const API_BASE = 'http://localhost:8000/api';
    
    window.SimilarCases = {
      
      // Fetch and display similar cases
      fetchSimilarCases: async function(imageFile) {
        try {
          const formData = new FormData();
          formData.append('file', imageFile);
          formData.append('k', 5);  // Top 5
          
          const response = await fetch(`${API_BASE}/similar/find`, {
            method: 'POST',
            body: formData
          });
          
          if (!response.ok) {
            throw new Error('Failed to fetch similar cases');
          }
          
          const data = await response.json();
          
          this.renderCases(data.similar_cases, data.search_time_ms);
          
        } catch (error) {
          console.error('Error fetching similar cases:', error);
          this.renderError(error.message);
        }
      },
      
      // Render cases grid
      renderCases: function(cases, searchTime) {
        const container = document.getElementById('similarPanel');
        
        if (!container) {
          console.error('Similar panel not found');
          return;
        }
        
        container.innerHTML = `
          <div class="similar-header">
            <h2>Similar Cases</h2>
            <span class="search-time">Search time: ${searchTime.toFixed(1)}ms</span>
          </div>
          
          <div class="similar-grid">
            ${cases.map(case_ => this.renderCaseCard(case_)).join('')}
          </div>
        `;
      },
      
      // Render single case card
      renderCaseCard: function(case_) {
        const similarityPercent = (case_.similarity_score * 100).toFixed(1);
        const statusColor = case_.has_tumor ? '#ff5252' : '#00c853';
        const statusText = case_.has_tumor ? 'Tumor Detected' : 'No Tumor';
        
        return `
          <div class="case-card" data-case-id="${case_.case_id}">
            <div class="case-rank">#{case_.rank}</div>
            
            <div class="case-thumbnail">
              ${case_.thumbnail ? `
                <img src="${case_.thumbnail}" alt="Case ${case_.case_id}"/>
              ` : `
                <div class="no-thumbnail">No Image</div>
              `}
            </div>
            
            <div class="case-info">
              <div class="similarity-score">
                <span class="label">Similarity:</span>
                <span class="value">${similarityPercent}%</span>
                <div class="score-bar">
                  <div class="score-fill" style="width: ${similarityPercent}%"></div>
                </div>
              </div>
              
              <div class="case-status" style="color: ${statusColor}">
                <span class="status-dot" style="background: ${statusColor}"></span>
                ${statusText}
              </div>
              
              <div class="case-meta">
                <div class="meta-item">
                  <span class="meta-label">Source:</span>
                  <span class="meta-value">${case_.source}</span>
                </div>
                <div class="meta-item">
                  <span class="meta-label">Patient:</span>
                  <span class="meta-value">${case_.patient_id}</span>
                </div>
              </div>
            </div>
            
            <button class="btn-view-case" onclick="SimilarCases.viewCaseDetails(${case_.case_id})">
              View Details
            </button>
          </div>
        `;
      },
      
      // View case details
      viewCaseDetails: async function(caseId) {
        try {
          const response = await fetch(`${API_BASE}/similar/case/${caseId}`);
          
          if (!response.ok) {
            throw new Error('Failed to fetch case details');
          }
          
          const caseData = await response.json();
          
          this.showCaseModal(caseData);
          
        } catch (error) {
          console.error('Error fetching case details:', error);
          alert('Failed to load case details');
        }
      },
      
      // Show case details modal
      showCaseModal: function(caseData) {
        // Create modal
        const modal = document.createElement('div');
        modal.className = 'case-modal';
        modal.innerHTML = `
          <div class="modal-overlay" onclick="this.parentElement.remove()"></div>
          <div class="modal-content">
            <div class="modal-header">
              <h2>Case #${caseData.case_id} Details</h2>
              <button class="btn-close" onclick="this.closest('.case-modal').remove()">×</button>
            </div>
            
            <div class="modal-body">
              <div class="case-image-large">
                ${caseData.image_base64 ? `
                  <img src="${caseData.image_base64}" alt="Case ${caseData.case_id}"/>
                ` : `
                  <div class="no-image">Image not available</div>
                `}
              </div>
              
              <div class="case-details">
                <div class="detail-row">
                  <span class="detail-label">Filename:</span>
                  <span class="detail-value">${caseData.filename}</span>
                </div>
                <div class="detail-row">
                  <span class="detail-label">Tumor:</span>
                  <span class="detail-value">${caseData.has_tumor ? 'Detected' : 'Not Detected'}</span>
                </div>
                <div class="detail-row">
                  <span class="detail-label">Source:</span>
                  <span class="detail-value">${caseData.source}</span>
                </div>
                <div class="detail-row">
                  <span class="detail-label">Patient ID:</span>
                  <span class="detail-value">${caseData.patient_id}</span>
                </div>
                ${caseData.label !== 'unknown' ? `
                  <div class="detail-row">
                    <span class="detail-label">Label:</span>
                    <span class="detail-value">${caseData.label}</span>
                  </div>
                ` : ''}
              </div>
            </div>
          </div>
        `;
        
        document.body.appendChild(modal);
      },
      
      // Render error
      renderError: function(message) {
        const container = document.getElementById('similarPanel');
        
        if (container) {
          container.innerHTML = `
            <div class="error-message">
              <div class="error-icon">⚠</div>
              <p>Failed to load similar cases</p>
              <p class="error-detail">${message}</p>
            </div>
          `;
        }
      }
      
    };
    
  })();