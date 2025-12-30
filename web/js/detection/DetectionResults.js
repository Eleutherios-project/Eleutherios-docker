/**
 * Detection Results Display
 * Renders detection analysis results with scores, signals, and findings
 */

export class DetectionResults {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.currentResults = null;
        this.expandedSignals = new Set();
    }
    
    /**
     * Render detection results
     * @param {object} results - Detection results from API
     */
    render(results) {
        if (!this.container) {
            console.error('DetectionResults: Container not found');
            return;
        }
        
        this.currentResults = results;
        const mode = results._meta?.mode || 'suppression';
        
        const html = `
            <div class="detection-results" data-mode="${mode}">
                ${this.renderHeader(results, mode)}
                ${this.renderOverallScore(results, mode)}
                ${this.renderSignalBreakdown(results, mode)}
                ${this.renderKeyFindings(results)}
                ${this.renderRecommendedActions(results, mode)}
            </div>
        `;
        
        this.container.innerHTML = html;
        this.container.style.display = 'block';
        this.attachEventListeners();
    }
    
    renderHeader(results, mode) {
        const modeConfig = {
            suppression: { icon: '🛡️', label: 'Suppression Detection', color: '#ff4444' },
            coordination: { icon: '🔗', label: 'Coordination Detection', color: '#ff9800' },
            anomaly: { icon: '🌍', label: 'Anomaly Detection', color: '#9c27b0' }
        };
        
        const config = modeConfig[mode] || modeConfig.suppression;
        const query = results.topic || results.pattern || results.query || results._meta?.query || 'Unknown';
        
        return `
            <div class="results-header" style="border-left-color: ${config.color}">
                <div class="header-title">
                    <span class="header-icon">${config.icon}</span>
                    <h2>${config.label} Results</h2>
                </div>
                <div class="header-query">
                    <span class="query-label">Query:</span>
                    <span class="query-text">"${this.escapeHtml(query)}"</span>
                </div>
                <div class="header-actions">
                    <button class="btn-action" id="export-results" title="Export results">
                        Export Results
                    </button>
                    <button class="btn-action" id="try-another-mode" title="Try another detection mode">
                        Try Another Mode
                    </button>
                    <button class="btn-action" id="new-query" title="Run new query">
                        New Query
                    </button>
                </div>
            </div>
        `;
    }
    
    renderOverallScore(results, mode) {
        const scoreKey = mode === 'suppression' ? 'suppression_score' :
                        mode === 'coordination' ? 'coordination_score' :
                        'anomaly_score';
        
        const score = results[scoreKey] || 0;
        const confidence = results.confidence || 0;
        const claimsAnalyzed = results.claims_analyzed || results.locations_analyzed || results.culture_count || 0;
        const interpretation = results.interpretation || 'No interpretation available';
        
        const scoreColor = this.getScoreColor(score);
        const scoreLabel = this.getScoreLabel(score);
        
        return `
            <div class="overall-score">
                <div class="score-header">
                    <h3>${mode.toUpperCase()} SCORE</h3>
                </div>
                <div class="score-display">
                    <div class="score-value" style="color: ${scoreColor}">
                        ${score.toFixed(2)}
                    </div>
                    <div class="score-bar">
                        <div class="score-fill" style="width: ${score * 100}%; background: ${scoreColor}"></div>
                    </div>
                    <div class="score-label" style="color: ${scoreColor}">
                        ${scoreLabel}
                    </div>
                </div>
                <div class="score-meta">
                    <span class="meta-item">
                        <strong>Confidence:</strong> ${(confidence * 100).toFixed(0)}%
                    </span>
                    <span class="meta-item">
                        <strong>Analyzed:</strong> ${claimsAnalyzed} ${mode === 'anomaly' ? 'cultures' : 'claims'}
                    </span>
                </div>
                <div class="score-interpretation">
                    ${this.escapeHtml(interpretation)}
                </div>
            </div>
        `;
    }
    
    renderSignalBreakdown(results, mode) {
        const signals = results.signals || (results.top_anomaly && results.top_anomaly.signals) || {};
        const signalEntries = Object.entries(signals);
        
        if (signalEntries.length === 0) {
            return '<div class="no-signals">No signal data available</div>';
        }
        
        return `
            <div class="signal-breakdown">
                <h3>Signal Breakdown</h3>
                <div class="signals-list">
                    ${signalEntries.map(([key, signal]) => 
                        this.renderSignal(key, signal)
                    ).join('')}
                </div>
            </div>
        `;
    }
    
    renderSignal(key, signal) {
        const score = signal.score || 0;
        const label = this.formatSignalLabel(key);
        const isExpanded = this.expandedSignals.has(key);
        const scoreColor = this.getScoreColor(score);
        
        return `
            <div class="signal-item ${isExpanded ? 'expanded' : ''}" data-signal="${key}">
                <div class="signal-header" data-signal-key="${key}">
                    <span class="signal-label">${label}</span>
                    <div class="signal-score-container">
                        <span class="signal-score" style="color: ${scoreColor}">
                            ${score.toFixed(2)}
                        </span>
                        <div class="signal-bar-mini">
                            <div class="signal-fill-mini" style="width: ${score * 100}%; background: ${scoreColor}"></div>
                        </div>
                    </div>
                    <button class="signal-expand-btn">${isExpanded ? '−' : '+'}</button>
                </div>
                ${isExpanded ? this.renderSignalDetails(key, signal) : ''}
            </div>
        `;
    }
    
    renderSignalDetails(key, signal) {
        const interpretation = signal.interpretation || 'No details available';
        const evidence = signal.evidence || {};
        
        return `
            <div class="signal-details">
                <p class="signal-interpretation">${this.escapeHtml(interpretation)}</p>
                ${this.renderEvidence(evidence)}
            </div>
        `;
    }
    
    renderEvidence(evidence) {
        if (!evidence || Object.keys(evidence).length === 0) {
            return '';
        }
        
        const items = [];
        for (const [key, value] of Object.entries(evidence)) {
            if (typeof value === 'number') {
                items.push(`<li><strong>${this.formatKey(key)}:</strong> ${value.toFixed(2)}</li>`);
            } else if (typeof value === 'string') {
                items.push(`<li><strong>${this.formatKey(key)}:</strong> ${this.escapeHtml(value)}</li>`);
            } else if (Array.isArray(value)) {
                items.push(`<li><strong>${this.formatKey(key)}:</strong> ${value.length} items</li>`);
            }
        }
        
        if (items.length === 0) return '';
        
        return `
            <div class="signal-evidence">
                <strong>Evidence:</strong>
                <ul>${items.join('')}</ul>
            </div>
        `;
    }
    
    renderKeyFindings(results) {
        const findings = results.key_findings || [];
        
        if (findings.length === 0) {
            return '';
        }
        
        return `
            <div class="key-findings">
                <h3>Key Findings</h3>
                <ol class="findings-list">
                    ${findings.map(finding => `
                        <li class="finding-item">${this.escapeHtml(finding)}</li>
                    `).join('')}
                </ol>
            </div>
        `;
    }
    
    renderRecommendedActions(results, mode) {
        const actions = this.getRecommendedActions(mode, results);
        
        return `
            <div class="recommended-actions">
                <h3>Recommended Actions</h3>
                <ul class="actions-list">
                    ${actions.map(action => `
                        <li class="action-item">${this.escapeHtml(action)}</li>
                    `).join('')}
                </ul>
                <div class="action-buttons">
                    ${this.getActionButtons(mode)}
                </div>
            </div>
        `;
    }
    
    getRecommendedActions(mode, results) {
        const score = results[`${mode}_score`] || 0;
        
        if (mode === 'suppression' && score > 0.7) {
            return [
                'Investigate suppressed primary research directly',
                'Document coordination patterns for evidence',
                'Cross-reference with coordination detection',
                'Review credential inversion cases',
                'Export full report for further analysis'
            ];
        } else if (mode === 'coordination' && score > 0.7) {
            return [
                'Identify sources of coordinated messaging',
                'Track temporal patterns and publication bursts',
                'Document language similarity patterns',
                'Investigate funding sources',
                'Export evidence for documentation'
            ];
        } else if (mode === 'anomaly' && score > 0.7) {
            return [
                'Research individual cultural instances',
                'Document shared pattern elements',
                'Investigate potential contact routes',
                'Consider alternative explanations',
                'Export findings for further study'
            ];
        } else {
            return [
                'No strong patterns detected',
                'Try different search terms',
                'Consider alternative detection modes',
                'Review standard search results'
            ];
        }
    }
    
    getActionButtons(mode) {
        if (mode === 'suppression') {
            return `
                <button class="btn-action-primary" id="run-coordination">
                    Run Coordination Detection
                </button>
                <button class="btn-action-secondary" id="export-report">
                    Export Full Report
                </button>
            `;
        } else if (mode === 'coordination') {
            return `
                <button class="btn-action-primary" id="run-suppression">
                    Run Suppression Detection
                </button>
                <button class="btn-action-secondary" id="export-report">
                    Export Full Report
                </button>
            `;
        } else {
            return `
                <button class="btn-action-primary" id="view-cultures">
                    View All Cultures
                </button>
                <button class="btn-action-secondary" id="export-report">
                    Export Full Report
                </button>
            `;
        }
    }
    
    attachEventListeners() {
        // Signal expand/collapse
        document.querySelectorAll('.signal-header').forEach(header => {
            header.addEventListener('click', () => {
                const signalKey = header.dataset.signalKey;
                this.toggleSignal(signalKey);
            });
        });
        
        // Export results
        document.getElementById('export-results')?.addEventListener('click', () => {
            this.exportResults();
        });
        
        // Try another mode
        document.getElementById('try-another-mode')?.addEventListener('click', () => {
            this.onTryAnotherMode?.();
        });
        
        // New query
        document.getElementById('new-query')?.addEventListener('click', () => {
            this.onNewQuery?.();
        });
        
        // Action buttons
        document.getElementById('run-coordination')?.addEventListener('click', () => {
            this.onRunMode?.('coordination');
        });
        
        document.getElementById('run-suppression')?.addEventListener('click', () => {
            this.onRunMode?.('suppression');
        });
        
        document.getElementById('view-cultures')?.addEventListener('click', () => {
            this.onViewCultures?.();
        });
        
        document.getElementById('export-report')?.addEventListener('click', () => {
            this.exportResults();
        });
    }
    
    toggleSignal(signalKey) {
        if (this.expandedSignals.has(signalKey)) {
            this.expandedSignals.delete(signalKey);
        } else {
            this.expandedSignals.add(signalKey);
        }
        
        // Re-render to show/hide details
        if (this.currentResults) {
            this.render(this.currentResults);
        }
    }
    
    exportResults() {
        if (!this.currentResults) return;
        
        const data = JSON.stringify(this.currentResults, null, 2);
        const blob = new Blob([data], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `detection-results-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }
    
    hide() {
        if (this.container) {
            this.container.style.display = 'none';
            this.container.innerHTML = '';
        }
    }
    
    // Helper methods
    
    getScoreColor(score) {
        if (score >= 0.8) return '#ff4444'; // Strong - Red
        if (score >= 0.6) return '#ff9800'; // Moderate - Orange
        if (score >= 0.4) return '#ffc107'; // Weak - Yellow
        return '#4caf50'; // Minimal - Green
    }
    
    getScoreLabel(score) {
        if (score >= 0.8) return 'STRONG';
        if (score >= 0.6) return 'MODERATE';
        if (score >= 0.4) return 'WEAK';
        if (score >= 0.2) return 'MINIMAL';
        return 'NONE';
    }
    
    formatSignalLabel(key) {
        return key
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
    
    formatKey(key) {
        return key
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    // Callbacks
    setTryAnotherModeCallback(callback) {
        this.onTryAnotherMode = callback;
    }
    
    setNewQueryCallback(callback) {
        this.onNewQuery = callback;
    }
    
    setRunModeCallback(callback) {
        this.onRunMode = callback;
    }
    
    setViewCulturesCallback(callback) {
        this.onViewCultures = callback;
    }
}
