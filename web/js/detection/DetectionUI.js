/**
 * Detection UI - Main Integration Module
 * Coordinates all detection UI components
 */

import { DetectionModeSelector } from './DetectionModeSelector.js';
import { DetectionRunner, DetectionLoadingState } from './DetectionRunner.js';
import { DetectionResults } from './DetectionResults.js';
import { GraphHighlighter } from './GraphHighlighter.js';

export class DetectionUI {
    constructor(config = {}) {
        // Configuration
        this.config = {
            modeSelectorId: config.modeSelectorId || 'detection-mode-selector',
            loadingStateId: config.loadingStateId || 'detection-loading-state',
            resultsContainerId: config.resultsContainerId || 'detection-results',
            searchInputId: config.searchInputId || 'search-query',
            searchButtonId: config.searchButtonId || 'search-button',
            graphRenderer: config.graphRenderer || null
        };
        
        // Components
        this.modeSelector = null;
        this.runner = null;
        this.loadingState = null;
        this.results = null;
        this.graphHighlighter = null;
        
        // State
        this.currentMode = 'standard';
        this.lastQuery = '';
        this.lastResults = null;
        
        this.init();
    }
    
    init() {
        console.log('Initializing Detection UI...');
        
        // Initialize mode selector
        this.modeSelector = new DetectionModeSelector(
            this.config.modeSelectorId,
            (mode) => this.onModeChange(mode)
        );
        
        // Initialize runner
        this.runner = new DetectionRunner();
        this.runner.setProgressCallback((phase, progress, message) => {
            this.onProgress(phase, progress, message);
        });
        this.runner.setCompleteCallback((result) => {
            this.onComplete(result);
        });
        this.runner.setErrorCallback((error) => {
            this.onError(error);
        });
        this.runner.setCancelledCallback(() => {
            this.onCancelled();
        });
        
        // Initialize loading state
        this.loadingState = new DetectionLoadingState(this.config.loadingStateId);
        this.loadingState.setCancelCallback(() => {
            this.runner.cancel();
        });
        
        // Initialize results display
        this.results = new DetectionResults(this.config.resultsContainerId);
        this.results.setTryAnotherModeCallback(() => {
            this.showModeSelector();
        });
        this.results.setNewQueryCallback(() => {
            this.focusSearchInput();
        });
        this.results.setRunModeCallback((mode) => {
            this.switchModeAndRun(mode);
        });
        
        // Initialize graph highlighter
        if (this.config.graphRenderer) {
            this.graphHighlighter = new GraphHighlighter(this.config.graphRenderer);
        }
        
        // Wire up search button
        this.setupSearchButton();
        
        console.log('Detection UI initialized successfully');
    }
    
    setupSearchButton() {
        const searchButton = document.getElementById(this.config.searchButtonId);
        const searchInput = document.getElementById(this.config.searchInputId);
        
        if (searchButton) {
            searchButton.addEventListener('click', () => {
                this.executeSearch();
            });
        }
        
        if (searchInput) {
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.executeSearch();
                }
            });
        }
    }
    
    onModeChange(mode) {
        this.currentMode = mode.id;
        
        // Update search UI
        this.updateSearchUI(mode);
        
        // Clear previous results if switching from standard mode
        if (this.currentMode === 'standard') {
            this.results.hide();
            if (this.graphHighlighter) {
                this.graphHighlighter.clearHighlights();
            }
        }
    }
    
    updateSearchUI(mode) {
        const searchInput = document.getElementById(this.config.searchInputId);
        const searchButton = document.getElementById(this.config.searchButtonId);
        
        if (searchInput) {
            searchInput.placeholder = mode.placeholder;
        }
        
        if (searchButton) {
            searchButton.textContent = mode.buttonText;
            searchButton.style.background = mode.id === 'standard' ? 
                'var(--accent-color)' : mode.color;
        }
    }
    
    executeSearch() {
        const searchInput = document.getElementById(this.config.searchInputId);
        if (!searchInput) return;
        
        const query = searchInput.value.trim();
        if (!query) {
            alert('Please enter a search query');
            return;
        }
        
        this.lastQuery = query;
        
        // Route to appropriate handler
        if (this.currentMode === 'standard') {
            this.executeStandardSearch(query);
        } else {
            this.executeDetection(this.currentMode, query);
        }
    }
    
    executeStandardSearch(query) {
        // Show pattern search results area (in case it was hidden by detection)
        const patternResults = document.getElementById('pattern-search-results');
        if (patternResults) patternResults.style.display = 'block';

        // Delegate to existing search functionality
        if (this.config.onStandardSearch) {
            this.config.onStandardSearch(query);
        } else {
            console.log('Standard search:', query);
        }
    }
    
    async executeDetection(mode, query) {
        try {
            // Hide previous results
            this.results.hide();
            // Hide pattern search results area during detection
            const patternResults = document.getElementById('pattern-search-results');
            if (patternResults) patternResults.style.display = 'none';
            // Show loading state
            this.loadingState.show('searching', 0, 'Initializing...');
            
            // Run detection
            const result = await this.runner.runDetection(mode, query);
            
            // Results will be handled by onComplete callback
            
        } catch (error) {
            // Errors will be handled by onError callback
            console.error('Detection failed:', error);
        }
    }
    
    onProgress(phase, progress, message) {
        this.loadingState.show(phase, progress, message);
    }
    
    onComplete(result) {
        // Hide loading
        this.loadingState.hide();
        
        // Store results
        this.lastResults = result;
        
        // Show results
        this.results.render(result);
        
        // Apply graph highlights
        if (this.graphHighlighter && result._meta) {
            this.graphHighlighter.applyHighlights(result, result._meta.mode);
        }
        
        // Scroll to results
        const resultsContainer = document.getElementById(this.config.resultsContainerId);
        if (resultsContainer) {
            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }
    
    onError(error) {
        // Hide loading
        this.loadingState.hide();
        
        // Show error
        const resultsContainer = document.getElementById(this.config.resultsContainerId);
        if (resultsContainer) {
            resultsContainer.innerHTML = `
                <div class="detection-error">
                    <div class="error-icon">❌</div>
                    <h3>Detection Failed</h3>
                    <p class="error-message">${this.escapeHtml(error.message)}</p>
                    <div class="error-actions">
                        <button class="btn-action" onclick="location.reload()">Retry</button>
                        <button class="btn-action" id="back-to-search">Back to Search</button>
                    </div>
                </div>
            `;
            
            resultsContainer.style.display = 'block';
            
            document.getElementById('back-to-search')?.addEventListener('click', () => {
                this.results.hide();
                this.focusSearchInput();
            });
        }
    }
    
    onCancelled() {
        // Hide loading
        this.loadingState.hide();
        
        // Show notification
        this.showNotification('Detection cancelled', 'info');
    }
    
    switchModeAndRun(mode) {
        // Switch mode
        this.modeSelector.selectMode(mode);
        
        // Re-run with last query
        if (this.lastQuery) {
            this.executeDetection(mode, this.lastQuery);
        }
    }
    
    showModeSelector() {
        const container = document.getElementById(this.config.modeSelectorId);
        if (container) {
            container.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }
    
    focusSearchInput() {
        const searchInput = document.getElementById(this.config.searchInputId);
        if (searchInput) {
            searchInput.focus();
            searchInput.select();
        }
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem 1.5rem;
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            z-index: 10000;
            animation: slideIn 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
    
    // Public API
    
    /**
     * Get current detection mode
     */
    getMode() {
        return this.currentMode;
    }
    
    /**
     * Set detection mode
     */
    setMode(mode) {
        this.modeSelector.selectMode(mode);
    }
    
    /**
     * Run detection programmatically
     */
    async runDetection(mode, query) {
        this.setMode(mode);
        return this.executeDetection(mode, query);
    }
    
    /**
     * Get last results
     */
    getLastResults() {
        return this.lastResults;
    }
    
    /**
     * Clear detection highlights
     */
    clearHighlights() {
        if (this.graphHighlighter) {
            this.graphHighlighter.clearHighlights();
        }
    }
    
    /**
     * Toggle highlights on/off
     */
    toggleHighlights(enabled) {
        if (this.graphHighlighter) {
            this.graphHighlighter.toggleHighlights(enabled);
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}
    /* =============================================================================
       Detection Placeholder Styles
       ============================================================================= */

    .detection-placeholder {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 300px;
        padding: 40px 20px;
    }

    .placeholder-content {
        text-align: center;
        max-width: 400px;
    }

    .placeholder-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.6;
    }

    .placeholder-title {
        font-size: 1.3rem;
        color: var(--text-primary, #e0e0e0);
        margin-bottom: 0.5rem;
    }

    .placeholder-subtitle {
        color: var(--text-secondary, #a0a0a0);
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .mode-highlight {
        color: var(--accent-color, #00ccff);
        font-weight: 600;
    }

    /* =============================================================================
       Narrative Section Styles
       ============================================================================= */

    .narrative-section {
        background: var(--card-bg, #1c2434);
        border: 1px solid var(--border-color, #2a3a5a);
        border-left: 4px solid var(--accent-color, #00ccff);
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1.5rem 0;
    }

    .narrative-section h4 {
        color: var(--accent-color, #00ccff);
        margin: 0 0 0.75rem 0;
        font-size: 1rem;
    }

    .narrative-content {
        color: var(--text-primary, #e0e0e0);
        line-height: 1.7;
        font-size: 0.95rem;
    }

    .narrative-content strong {
        color: var(--accent-color, #00ccff);
    }

    /* =============================================================================
       Claim Numbering Styles
       ============================================================================= */

    .result-item,
    .claim-item {
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }

    .claim-number {
        color: var(--accent-color, #00ccff);
        font-weight: 600;
        font-size: 0.9rem;
        min-width: 28px;
        flex-shrink: 0;
        padding-top: 2px;
    }

    .result-content,
    .claim-content {
        flex: 1;
        min-width: 0; /* Allow text truncation */
    }

    /* =============================================================================
       Score Display Enhancements
       ============================================================================= */

    .score-display {
        text-align: center;
        padding: 1.5rem;
        margin: 1rem 0;
        background: var(--card-bg, #1c2434);
        border-radius: 8px;
        border: 1px solid var(--border-color, #2a3a5a);
    }

    .score-value {
        font-size: 3rem;
        font-weight: bold;
        color: var(--score-color, #00ccff);
        font-family: 'Courier New', monospace;
    }

    .score-bar {
        height: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        margin: 1rem auto;
        max-width: 300px;
        overflow: hidden;
    }

    .score-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    .score-interpretation {
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .score-confidence {
        color: var(--text-secondary, #a0a0a0);
        font-size: 0.9rem;
    }

    /* =============================================================================
       Signal Breakdown Improvements
       ============================================================================= */

    .signal-breakdown {
        margin: 1.5rem 0;
        padding: 1rem;
        background: var(--card-bg, #1c2434);
        border-radius: 8px;
        border: 1px solid var(--border-color, #2a3a5a);
    }

    .signal-breakdown h4 {
        color: var(--text-primary, #e0e0e0);
        margin: 0 0 1rem 0;
        font-size: 1rem;
    }

    .signal-item {
        margin-bottom: 0.75rem;
        cursor: pointer;
        padding: 0.5rem;
        border-radius: 4px;
        transition: background 0.2s;
    }

    .signal-item:hover {
        background: rgba(0, 204, 255, 0.1);
    }

    .signal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.25rem;
    }

    .signal-name {
        color: var(--text-secondary, #a0a0a0);
        font-size: 0.9rem;
    }

    .signal-score {
        font-weight: 600;
        font-family: 'Courier New', monospace;
    }

    .signal-bar {
        height: 4px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 2px;
        overflow: hidden;
    }

    .signal-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 0.3s ease;
    }

    /* =============================================================================
       Key Findings Section
       ============================================================================= */

    .key-findings {
        margin: 1.5rem 0;
        padding: 1rem;
        background: var(--card-bg, #1c2434);
        border-radius: 8px;
        border: 1px solid var(--border-color, #2a3a5a);
    }

    .key-findings h4 {
        color: var(--text-primary, #e0e0e0);
        margin: 0 0 1rem 0;
        font-size: 1rem;
    }

    .finding-item {
        display: flex;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
        padding: 0.5rem;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 4px;
    }

    .finding-number {
        color: var(--accent-color, #00ccff);
        font-weight: 600;
        min-width: 20px;
    }

    .finding-text {
        color: var(--text-primary, #e0e0e0);
        line-height: 1.5;
    }

    /* =============================================================================
       Affected Claims Section
       ============================================================================= */

    .affected-claims {
        margin: 1.5rem 0;
    }

    .affected-claims h4 {
        color: var(--text-primary, #e0e0e0);
        margin: 0 0 1rem 0;
        font-size: 1rem;
    }

    .claims-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .claim-item {
        padding: 0.75rem;
        background: var(--card-bg, #1c2434);
        border: 1px solid var(--border-color, #2a3a5a);
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.2s;
    }

    .claim-item:hover {
        border-color: var(--accent-color, #00ccff);
        transform: translateX(4px);
    }

    .claim-text {
        color: var(--text-primary, #e0e0e0);
        line-height: 1.5;
    }

    .claim-score {
        margin-top: 0.5rem;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .claims-more {
        text-align: center;
        color: var(--text-secondary, #a0a0a0);
        padding: 0.75rem;
        font-style: italic;
    }

    /* =============================================================================
       Results Header
       ============================================================================= */

    .results-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border-color, #2a3a5a);
    }

    .results-header h3 {
        margin: 0;
        color: var(--text-primary, #e0e0e0);
    }

    .results-count {
        color: var(--text-secondary, #a0a0a0);
        font-size: 0.9rem;
    }

    .detection-header {
        border-left: 4px solid var(--mode-color, #00ccff);
        padding-left: 1rem;
    }

    .results-actions {
        display: flex;
        gap: 0.5rem;
    }

    .btn-small {
        padding: 0.4rem 0.75rem;
        font-size: 0.85rem;
        background: rgba(0, 204, 255, 0.2);
        border: 1px solid var(--accent-color, #00ccff);
        color: var(--accent-color, #00ccff);
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.2s;
    }

    .btn-small:hover {
        background: var(--accent-color, #00ccff);
        color: var(--primary-bg, #0a0e17);
    }

    /* =============================================================================
       Detection Error
       ============================================================================= */

    .detection-error {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem;
        background: rgba(244, 67, 54, 0.1);
        border: 1px solid #f44336;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .error-icon {
        font-size: 1.5rem;
    }

    .error-text {
        flex: 1;
        color: #f44336;
    }

    .retry-btn {
        padding: 0.5rem 1rem;
        background: #f44336;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }

    .retry-btn:hover {
        background: #d32f2f;
    }
// CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .detection-error {
        text-align: center;
        padding: 3rem 2rem;
        background: var(--card-bg);
        border-radius: 12px;
        border: 2px solid var(--score-strong);
    }
    
    .error-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .error-message {
        color: var(--text-secondary);
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    .error-actions {
        display: flex;
        justify-content: center;
        gap: 1rem;
    }
`;
document.head.appendChild(style);
