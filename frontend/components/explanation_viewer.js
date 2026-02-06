(function ExplanationViewer() {
  
    window.ExplanationViewer = {
      
      // Initialize viewer
      init: function() {
        console.log('ExplanationViewer initialized');
        
        // Add event listeners
        this.attachEventListeners();
      },
      
      // Attach event listeners
      attachEventListeners: function() {
        // Tab switching
        const pills = document.querySelectorAll('.pill');
        pills.forEach(pill => {
          pill.addEventListener('click', (e) => {
            this.switchTab(e.target.dataset.tab);
          });
        });
        
        // Compare toggle
        const compareBtn = document.getElementById('btnCompare');
        if (compareBtn) {
          compareBtn.addEventListener('click', () => {
            this.toggleCompareMode();
          });
        }
      },
      
      // Switch tab
      switchTab: function(tab) {
        // Hide all panels
        document.querySelectorAll('.panel, .xai-panel, .similar-panel, .info-panel')
          .forEach(panel => panel.style.display = 'none');
        
        // Show selected panel
        if (tab === 'scan') {
          document.querySelectorAll('.panel').forEach(p => p.style.display = 'flex');
        } else if (tab === 'xai') {
          const xaiPanel = document.getElementById('xaiPanel');
          if (xaiPanel) xaiPanel.style.display = 'block';
        } else if (tab === 'similar') {
          const similarPanel = document.getElementById('similarPanel');
          if (similarPanel) similarPanel.style.display = 'block';
        } else if (tab === 'info') {
          const infoPanel = document.getElementById('infoPanel');
          if (infoPanel) infoPanel.style.display = 'block';
        }
        
        // Update active pill
        document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
        document.querySelector(`.pill[data-tab="${tab}"]`)?.classList.add('active');
      },
      
      // Toggle compare mode
      toggleCompareMode: function() {
        const isCompare = document.body.classList.toggle('compare-mode');
        
        if (isCompare) {
          this.enableCompareMode();
        } else {
          this.disableCompareMode();
        }
      },
      
      // Enable compare mode
      enableCompareMode: function() {
        console.log('Compare mode enabled');
        // Implementation for side-by-side comparison
      },
      
      // Disable compare mode
      disableCompareMode: function() {
        console.log('Compare mode disabled');
      },
      
      // Export explanation
      exportExplanation: function(format = 'pdf') {
        console.log(`Exporting explanation as ${format}`);
        // Implementation for export functionality
        alert('Export feature coming soon!');
      }
      
    };
    
    // Auto-initialize on DOM ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => {
        window.ExplanationViewer.init();
      });
    } else {
      window.ExplanationViewer.init();
    }
    
  })();