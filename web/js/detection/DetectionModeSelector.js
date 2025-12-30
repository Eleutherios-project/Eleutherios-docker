/**
 * Detection Mode Selector (UPDATED v1.1)
 * Tab-style component for switching between detection modes
 * 
 * CHANGES from v1.0:
 * - Reordered modes array: Standard mode first, then detectors
 * - No logic changes, just array reordering
 * 
 * Date: November 23, 2025
 */

export class DetectionModeSelector {
    constructor(containerId, onModeChange) {
        this.container = document.getElementById(containerId);
        this.onModeChange = onModeChange;
        this.currentMode = 'standard'; // default mode
        
        // UPDATED: Mode order changed - Standard first, then detectors
        this.modes = [
            // STANDARD MODE FIRST
            {
                id: 'standard',
                label: 'Standard',
                icon: '🔍',
                description: 'Regular knowledge graph search',
                color: '#00ccff',
                placeholder: 'Search for claims, entities, or topics...',
                buttonText: 'Search',
                examples: ['COVID', 'climate', 'archaeology']
            },
            // THEN DETECTION MODES
            {
                id: 'suppression',
                label: 'Suppression',
                icon: '🛡️',
                description: 'Detect systematic suppression patterns',
                color: '#ff4444',
                placeholder: 'Enter topic to analyze for suppression...',
                buttonText: 'Detect Suppression',
                examples: ['ivermectin COVID', 'vaccine adverse events', 'lab leak theory']
            },
            {
                id: 'coordination',
                label: 'Coordination',
                icon: '🔗',
                description: 'Identify coordinated messaging campaigns',
                color: '#ff9800',
                placeholder: 'Enter topic to detect coordination...',
                buttonText: 'Detect Coordination',
                examples: ['horse dewormer', 'climate emergency', 'safe and effective']
            },
            {
                id: 'anomaly',
                label: 'Anomalies',
                icon: '🌍',
                description: 'Find cross-cultural patterns and anomalies',
                color: '#9c27b0',
                placeholder: 'Enter pattern to find anomalies...',
                buttonText: 'Find Anomalies',
                examples: ['flood mythology', 'pyramid construction', 'reed boat design']
            }
        ];
        
        this.render();
    }
    
    render() {
        if (!this.container) {
            console.error('DetectionModeSelector: Container not found');
            return;
        }
        
        const html = `
            <div class="detection-mode-selector">
                <div class="mode-buttons">
                    ${this.modes.map(mode => this.renderModeButton(mode)).join('')}
                </div>
                <div class="mode-description" id="mode-description">
                    ${this.getModeDescription(this.currentMode)}
                </div>
                <div class="mode-examples" id="mode-examples">
                    ${this.getModeExamples(this.currentMode)}
                </div>
            </div>
        `;
        
        this.container.innerHTML = html;
        this.attachEventListeners();
    }
    
    renderModeButton(mode) {
        const isActive = mode.id === this.currentMode;
        return `
            <button 
                class="mode-button ${isActive ? 'active' : ''}" 
                data-mode="${mode.id}"
                style="--mode-color: ${mode.color}"
                title="${mode.description}"
            >
                <span class="mode-icon">${mode.icon}</span>
                <span class="mode-label">${mode.label}</span>
            </button>
        `;
    }
    
    getModeDescription(modeId) {
        const mode = this.modes.find(m => m.id === modeId);
        return mode ? `
            <span class="mode-desc-icon">${mode.icon}</span>
            <span class="mode-desc-text">${mode.description}</span>
        ` : '';
    }
    
    getModeExamples(modeId) {
        const mode = this.modes.find(m => m.id === modeId);
        if (!mode || !mode.examples || mode.examples.length === 0) return '';
        
        return `
            <span class="examples-label">Examples:</span>
            ${mode.examples.map(example => 
                `<button class="example-button" data-example="${this.escapeHtml(example)}">${this.escapeHtml(example)}</button>`
            ).join('')}
        `;
    }
    
    attachEventListeners() {
        // Mode button clicks
        document.querySelectorAll('.mode-button').forEach(button => {
            button.addEventListener('click', () => {
                const mode = button.dataset.mode;
                this.selectMode(mode);
            });
        });
        
        // Example button clicks
        document.querySelectorAll('.example-button').forEach(button => {
            button.addEventListener('click', () => {
                const example = button.dataset.example;
                this.fillSearchWithExample(example);
            });
        });
    }
    
    selectMode(modeId) {
        if (this.currentMode === modeId) return;
        
        this.currentMode = modeId;
        
        // Update button states
        document.querySelectorAll('.mode-button').forEach(button => {
            if (button.dataset.mode === modeId) {
                button.classList.add('active');
            } else {
                button.classList.remove('active');
            }
        });
        
        // Update description
        const descContainer = document.getElementById('mode-description');
        if (descContainer) {
            descContainer.innerHTML = this.getModeDescription(modeId);
        }
        
        // Update examples
        const examplesContainer = document.getElementById('mode-examples');
        if (examplesContainer) {
            examplesContainer.innerHTML = this.getModeExamples(modeId);
            // Re-attach listeners for new example buttons
            examplesContainer.querySelectorAll('.example-button').forEach(button => {
                button.addEventListener('click', () => {
                    const example = button.dataset.example;
                    this.fillSearchWithExample(example);
                });
            });
        }
        
        // Notify parent
        if (this.onModeChange) {
            const mode = this.modes.find(m => m.id === modeId);
            this.onModeChange(mode);
        }
    }
    
    fillSearchWithExample(example) {
        // Find search input and fill it
        const searchInput = document.getElementById('search-query') || 
                          document.querySelector('input[type="text"]');
        if (searchInput) {
            searchInput.value = example;
            searchInput.focus();
        }
    }
    
    getCurrentMode() {
        return this.modes.find(m => m.id === this.currentMode);
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}
