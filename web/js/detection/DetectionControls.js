/**
 * Aegis Insight v2.0 - Detection Controls
 * UI components for detection mode selection and results display
 * 
 * TWO-STAGE PIPELINE:
 * 1. Pattern Search → Get semantically relevant claim IDs (uses PostgreSQL embeddings)
 * 2. Detection Analysis → Analyze those specific claims (uses Neo4j graph data)
 * 
 * This ensures detection uses the same semantic search as Pattern Search mode.
 */

import { CONFIG, getScoreColor, getScoreInterpretation } from '../config.js';

export class DetectionControls {
    constructor(options = {}) {
        this.containerId = options.containerId || '#detection-panel';
        this.apiBaseUrl = options.apiBaseUrl || 'http://localhost:8001';
        this.onDetectionComplete = options.onDetectionComplete || (() => {});
        this.onModeChange = options.onModeChange || (() => {});
        
        this.activeMode = 'standard';
        this.isLoading = false;
        this.lastResults = null;
        this.lastMode = null;
        this.lastClaimIds = null;  // Store claim IDs for debugging
        
        this.container = null;
        this.searchInput = null;
        this.searchButton = null;
    }

    /**
     * Initialize the detection controls
     */
    initialize(container) {
        this.container = container || document.querySelector(this.containerId);
        if (!this.container) {
            console.error('Detection controls container not found');
            return;
        }
        
        this.render();
        this.attachEventListeners();
    }

    /**
     * Render detection controls
     */
    render() {
        this.container.innerHTML = `
            <div class="detection-controls">
                <!-- Mode Selector -->
                <div class="detection-mode-selector">
                    ${this.renderModeButtons()}
                </div>
                
                <!-- Search Bar -->
                <div class="detection-search-bar">
                    <input type="text" 
                           id="detection-search-input"
                           class="detection-search-input"
                           placeholder="${CONFIG.DETECTION_MODES[this.activeMode].placeholder}">
                    <button id="detection-search-btn" class="detection-search-btn">
                        ${CONFIG.DETECTION_MODES[this.activeMode].buttonText}
                    </button>
                    <!-- Profile Selector (for detection modes) -->
                    <div class="detection-profile-selector" id="profile-selector" style="display: ${this.activeMode === 'standard' ? 'none' : 'inline-flex'}; align-items: center; margin-left: 15px; gap: 5px;">
                        <label style="color: #888; font-size: 12px;">Profile:</label>
                        <select id="detection-profile-select" class="detection-profile-select" style="padding: 4px 8px; background: #1a1a2e; color: #fff; border: 1px solid #333; border-radius: 4px;">
                            <option value="default.json">Default - Balanced</option>
                            <option value="state_suppression.json">State Suppression</option>
                            <option value="academic_gatekeeping.json">Academic Gatekeeping</option>
                            <option value="modern_fact_check.json">Modern Fact-Check</option>
                            <option value="ideological_subversion.json">Ideological Subversion</option>
                        </select>
                    </div>
                </div>
                
                <!-- Helper Text -->
                <div class="detection-helper" id="detection-helper">
                    Examples: "Thomas Paine", "Spanish American War", "pyramid construction"
                </div>
                
                <!-- Claim Limit Control (moved below examples for better layout) -->
                <div class="claim-limit-control" style="display: flex; align-items: center; justify-content: flex-start; margin-top: 8px; gap: 8px;">
                    <label style="color: #888; font-size: 12px;">Claims to analyze:</label>
                    <button id="claim-limit-minus" class="btn-small" style="padding: 4px 8px; font-size: 14px;">-</button>
                    <input type="number" id="claim-limit-input" value="170" min="10" max="1000" step="5" style="width: 60px; text-align: center; padding: 4px; background: #1a1a2e; color: #fff; border: 1px solid #333; border-radius: 4px;">
                    <button id="claim-limit-plus" class="btn-small" style="padding: 4px 8px; font-size: 14px;">+</button>
                </div>
                
                <!-- Results Panel -->
                <div class="detection-results" id="detection-results" style="display: none;">
                    <!-- Populated when results arrive -->
                </div>
                
                <!-- Loading Overlay -->
                <div class="detection-loading" id="detection-loading" style="display: none;">
                    <div class="loading-spinner"></div>
                    <div class="loading-text" id="loading-text">Analyzing patterns...</div>
                    <div class="loading-subtext" id="loading-subtext">This may take 15-30 seconds</div>
                </div>
            </div>
        `;
        
        // Store references
        this.searchInput = this.container.querySelector('#detection-search-input');
        this.searchButton = this.container.querySelector('#detection-search-btn');
        this.profileSelect = this.container.querySelector('#detection-profile-select');
    }

    /**
     * Render mode selector buttons
     */
    renderModeButtons() {
        return Object.entries(CONFIG.DETECTION_MODES).map(([mode, config]) => `
            <button class="mode-button ${mode === this.activeMode ? 'active' : ''}" 
                    data-mode="${mode}"
                    style="--mode-color: ${config.color}">
                <span class="mode-icon">${config.icon}</span>
                <span class="mode-label">${config.name.replace(' Detection', '').replace(' Search', '')}</span>
            </button>
        `).join('');
    }

    /**
     * Attach event listeners
     */
    attachEventListeners() {
        // Mode buttons
        this.container.querySelectorAll('.mode-button').forEach(btn => {
            btn.addEventListener('click', () => this.setMode(btn.dataset.mode));
        });
        
        // Search button
        this.searchButton.addEventListener('click', () => this.executeSearch());
        
        // Enter key
        this.searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.executeSearch();
        });
    }

    /**
     * Set active detection mode
     */
    setMode(mode) {
        if (!CONFIG.DETECTION_MODES[mode]) return;
        
        this.activeMode = mode;
        const config = CONFIG.DETECTION_MODES[mode];
        
        // Update buttons
        this.container.querySelectorAll('.mode-button').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
        
        // Update search
        this.searchInput.placeholder = config.placeholder;
        this.searchButton.textContent = config.buttonText;
        this.searchButton.style.background = config.color;
        
        // Update helper
        const helper = this.container.querySelector('#detection-helper');
        if (mode === 'standard') {
            helper.style.display = 'none';
        } else {
            helper.style.display = 'block';
            helper.innerHTML = this.getExamplesForMode(mode);
        }
        
        // Show/hide profile selector (only for detection modes, not standard)
        const profileSelector = this.container.querySelector('#profile-selector');
        if (profileSelector) {
            profileSelector.style.display = mode === 'standard' ? 'none' : 'inline-flex';
        }
        
        // Handle results display based on mode state
        if (this.lastResults && this.lastMode === mode) {
            // Same mode with cached results - show them
            this.displayResults(this.lastResults, mode);
        } else if (mode !== 'standard') {
            // Detection mode without results - show placeholder
            this.showModePlaceholder(mode);
        } else {
            // Standard mode or no results - hide
            this.hideResults();
        }
        
        // Callback
        this.onModeChange(mode);
    }

    /**
     * Get example queries for mode
     */
    getExamplesForMode(mode) {
        const examples = {
            suppression: 'Examples: "Thomas Paine", "Smedley Butler", "alternative archaeology"',
            coordination: 'Examples: "Spanish American War", "Remember the Maine", "yellow journalism"',
            anomaly: '<span style="color: #ff9800;">Experimental Feature - Results may be unreliable</span><br/>Examples: "pyramid construction", "flood mythology", "ancient navigation"',
        };
        return examples[mode] || '';
    }

    /**
     * Execute search/detection
     */
    async executeSearch() {
        const query = this.searchInput.value.trim();
        if (!query) {
            this.searchInput.focus();
            return;
        }
        
        if (this.isLoading) return;
        
        this.showLoading();
        
        try {
            let results;
            
            if (this.activeMode === 'standard') {
                results = await this.executeStandardSearch(query);
            } else {
                // TWO-STAGE PIPELINE for detection modes
                results = await this.executeDetection(query, this.activeMode);
            }
            
            this.lastResults = results;
            this.lastMode = this.activeMode;
            this.displayResults(results, this.activeMode);
            
            // Unwrap nested result for graph highlighting
            // API returns { success, result: { affected_claims, ... } }
            const highlightData = results.result || results;
            this.onDetectionComplete(highlightData, this.activeMode);
            
        } catch (error) {
            console.error('Detection error:', error);
            this.displayError(error.message);
        } finally {
            this.hideLoading();
        }
    }

    /**
     * Execute standard search
     */
    async executeStandardSearch(query) {
        const response = await fetch(`${this.apiBaseUrl}/api/pattern-search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, limit: parseInt(document.getElementById('claim-limit-input')?.value) || 170 })
        });
        
        if (!response.ok) {
            throw new Error(`Search failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Execute detection analysis - TWO-STAGE PIPELINE
     * 
     * Stage 1: Pattern search to get semantically relevant claims (PostgreSQL embeddings)
     * Stage 2: Detection analysis on those specific claims (Neo4j graph data)
     * 
     * This ensures detection uses the same semantic search as Pattern Search mode.
     */
    async executeDetection(topic, mode) {
        // ============================================
        // STAGE 1: Pattern Search for Claim IDs
        // ============================================
        this.updateLoadingText('Finding relevant claims...', 'Stage 1 of 2: Semantic search');
        
        const searchResponse = await fetch(`${this.apiBaseUrl}/api/pattern-search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: topic, limit: parseInt(document.getElementById('claim-limit-input')?.value) || 170 })
        });
        
        if (!searchResponse.ok) {
            const error = await searchResponse.json().catch(() => ({}));
            throw new Error(error.detail || `Pattern search failed: ${searchResponse.status}`);
        }
        
        const searchResult = await searchResponse.json();
        
        // Extract claim IDs from pattern search results
        const claims = searchResult.claims || searchResult.results || [];
        const claimIds = claims
            .map(c => c.id || c.claim_id || c.elementId)
            .filter(Boolean);
        
        console.log(`[DetectionControls] Pattern search found ${claimIds.length} claims for ${mode} detection`);
        this.lastClaimIds = claimIds;  // Store for debugging
        
        if (claimIds.length === 0) {
            throw new Error('No claims found matching query. Try different search terms.');
        }
        
        // ============================================
        // STAGE 2: Detection Analysis on Those Claims
        // ============================================
        this.updateLoadingText(
            `Analyzing ${claimIds.length} claims for ${mode} patterns...`, 
            'Stage 2 of 2: Detection analysis'
        );
        
        const endpoint = `/api/detect/${mode}`;
        
        const response = await fetch(`${this.apiBaseUrl}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                topic,
                query: topic,      // Some detectors use 'query' instead of 'topic'
                claim_ids: claimIds,  // Pass semantic search results to detector
                profile: this.profileSelect?.value || 'state_suppression.json'  // Include selected profile
            })
        });
        
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `Detection failed: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Add claim count metadata if not present
        if (!result.claims_analyzed && claimIds.length > 0) {
            result.claims_analyzed = claimIds.length;
        }
        
        return result;
    }

    /**
     * Update loading text during two-stage pipeline
     */
    updateLoadingText(mainText, subText) {
        const loadingTextEl = this.container.querySelector('#loading-text');
        const loadingSubtextEl = this.container.querySelector('#loading-subtext');
        
        if (loadingTextEl) loadingTextEl.textContent = mainText;
        if (loadingSubtextEl) loadingSubtextEl.textContent = subText;
    }

    /**
     * Display detection results
     */
    displayResults(results, mode) {
        const resultsPanel = this.container.querySelector('#detection-results');
        
        // Handle nested results (suppression wraps in 'result')
        const data = results.result || results;
        
        if (mode === 'standard') {
            resultsPanel.innerHTML = this.renderStandardResults(data);
        } else {
            resultsPanel.innerHTML = this.renderDetectionResults(data, mode);
        }
        
        resultsPanel.style.display = 'block';
        
        // Attach result interaction handlers
        this.attachResultHandlers();
    }

    /**
     * Render standard search results
     */
    renderStandardResults(results) {
        const claims = results.claims || results.results || [];
        const synthesis = results.synthesis || '';
        
        return `
            <div class="results-header">
                <h3>🔍 Search Results</h3>
                <span class="results-count">${claims.length} claims found</span>
            </div>
            
            ${synthesis ? `
            <div class="synthesis-section" style="background: #1a1a2e; border-left: 3px solid #4a9eff; padding: 15px; margin: 10px 0; border-radius: 4px;">
                <h4 style="margin: 0 0 10px 0; color: #4a9eff;">Analysis</h4>
                <div class="synthesis-content" style="white-space: pre-wrap; line-height: 1.6;">${synthesis.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")}</div>
            </div>
            ` : ''}
            
            <div class="results-list">
                ${claims.map((claim, index) => `
                    <div class="result-item" data-id="${claim.id || claim.claim_id}">
                        <span class="claim-number">${index + 1}.</span>
                        <div class="result-content">
                            <div class="result-text">${this.truncate(claim.claim_text || claim.text, 150)}</div>
                            <div class="result-meta">
                                ${claim.confidence ? `<span class="meta-confidence">${(claim.confidence * 100).toFixed(0)}% confidence</span>` : ''}
                                ${claim.source_file ? `<span class="meta-source">${claim.source_file.split('/').pop()}</span>` : ''}
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    /**
     * Render detection results
     */
    renderDetectionResults(results, mode) {
        const score = results.suppression_score || results.coordination_score || results.anomaly_score || 0;
        const scoreColor = getScoreColor(score);
        const interpretation = getScoreInterpretation(score);
        const config = CONFIG.DETECTION_MODES[mode];
        
        // Use stored claim count if not in results
        const claimsAnalyzed = results.claims_analyzed || this.lastClaimIds?.length || 0;
        
        return `
            <div class="results-header detection-header" style="--mode-color: ${config.color}">
                <h3>${config.icon} ${config.name} Results</h3>
                <div class="results-actions">
                    <button class="btn-small" id="export-results-btn">📄 Export</button>
                    <button class="btn-small" id="try-another-btn">🔄 Try Another Mode</button>
                </div>
            </div>
            
            <!-- Overall Score -->
            <div class="score-display" style="--score-color: ${scoreColor}">
                <div class="score-value">${score.toFixed(2)}</div>
                <div class="score-bar">
                    <div class="score-fill" style="width: ${score * 100}%; background: ${scoreColor}"></div>
                </div>
                <div class="score-interpretation" style="color: ${scoreColor}">
                    ${interpretation} ${mode} pattern detected
                </div>
                <div class="score-confidence">
                    ${claimsAnalyzed} claims analyzed via semantic search
                </div>
            </div>
            
            <!-- Narrative Interpretation -->
            ${this.renderNarrativeInterpretation(results, mode, claimsAnalyzed)}
            
            <!-- Signal Breakdown -->
            ${this.renderSignalBreakdown(results, mode)}
            
            <!-- Key Findings -->
            ${this.renderKeyFindings(results, mode)}
            
            <!-- Affected Claims -->
            ${this.renderAffectedClaims(results, mode)}
        `;
    }

    /**
     * Render narrative interpretation paragraph
     */
    renderNarrativeInterpretation(results, mode, claimsAnalyzed) {
        const narrative = this.generateNarrative(results, mode, claimsAnalyzed);
        
        if (!narrative) return '';
        
        return `
            <div class="narrative-interpretation">
                <p>${narrative}</p>
            </div>
        `;
    }

    /**
     * Generate human-readable narrative based on scores
     */
    generateNarrative(results, mode, claimsAnalyzed) {
        const score = results.suppression_score || results.coordination_score || results.anomaly_score || 0;
        const signals = results.signals || {};
        
        let narrative = '';
        
        if (mode === 'suppression') {
            // Use level from API if available, otherwise calculate from score
            const level = results.level || (score >= 0.75 ? 'CRITICAL' : score >= 0.55 ? 'HIGH' : score >= 0.35 ? 'MODERATE' : 'LOW');
            
            if (level === 'CRITICAL' || score >= 0.75) {
                narrative = `Analysis reveals CRITICAL suppression patterns across ${claimsAnalyzed} claims. `;
            } else if (level === 'HIGH' || score >= 0.55) {
                narrative = `Analysis reveals HIGH-level suppression patterns across ${claimsAnalyzed} claims. `;
            } else if (level === 'MODERATE' || score >= 0.35) {
                narrative = `Analysis reveals MODERATE suppression patterns across ${claimsAnalyzed} claims. `;
            } else {
                narrative = `Analysis reveals LOW-level suppression patterns across ${claimsAnalyzed} claims. `;
            }
            
            // Add signal-specific details
            const meta = signals.meta_claim_density?.score || signals.meta_claim_density || 0;
            const isolation = signals.network_isolation?.score || signals.network_isolation || 0;
            const evidence = signals.evidence_avoidance?.score || signals.evidence_avoidance || 0;
            
            if (meta > 0.5) narrative += 'Significant meta-commentary present (claims about claims rather than evidence). ';
            else if (meta > 0.2) narrative += 'Some meta-commentary present. ';
            
            if (isolation > 0.5) narrative += 'Information appears isolated from mainstream discourse. ';
            else if (isolation > 0.2) narrative += 'Some isolation detected. ';
            
            if (evidence > 0.5) narrative += 'Evidence-based discussion appears suppressed. ';
            
            if (score < 0.3) narrative += 'Overall, information ecosystem appears relatively functional.';
            
        } else if (mode === 'coordination') {
            if (score >= 0.7) {
                narrative = `Analysis reveals HIGH coordination patterns suggesting manufactured consensus across ${claimsAnalyzed} claims. `;
            } else if (score >= 0.4) {
                narrative = `Analysis reveals MODERATE coordination patterns across ${claimsAnalyzed} claims. `;
            } else {
                narrative = `Analysis reveals LOW coordination patterns across ${claimsAnalyzed} claims. `;
            }
            
            const temporal = signals.temporal_clustering?.score || signals.temporal_clustering || 0;
            const linguistic = signals.linguistic_similarity?.score || signals.linguistic_similarity || 0;
            
            if (temporal > 0.5) narrative += 'Significant temporal clustering detected (synchronized messaging). ';
            if (linguistic > 0.5) narrative += 'High linguistic similarity suggests coordinated talking points. ';
            
        } else if (mode === 'anomaly') {
            if (score >= 0.7) {
                narrative = `Analysis reveals STRONG cross-cultural anomalies across ${claimsAnalyzed} claims. `;
            } else if (score >= 0.4) {
                narrative = `Analysis reveals MODERATE cross-cultural anomalies across ${claimsAnalyzed} claims. `;
            } else {
                narrative = `Analysis reveals WEAK cross-cultural anomalies across ${claimsAnalyzed} claims. `;
            }
            
            const geographic = signals.geographic_isolation?.score || signals.geographic_isolation || 0;
            const temporal = signals.temporal_impossibility?.score || signals.temporal_impossibility || 0;
            
            if (geographic > 0.5) narrative += 'Similar patterns found in geographically isolated cultures. ';
            if (temporal > 0.5) narrative += 'Temporal alignment suggests shared origin or transmission. ';
        }
        
        return narrative;
    }

    /**
     * Render signal breakdown
     */
    renderSignalBreakdown(results, mode) {
        const signals = results.signals || results.signal_breakdown || {};
        
        if (Object.keys(signals).length === 0) return '';
        
        return `
            <div class="signal-breakdown">
                <h4>Signal Breakdown</h4>
                ${Object.entries(signals).map(([name, data]) => {
                    const score = typeof data === 'number' ? data : (data.score || 0);
                    const color = getScoreColor(score);
                    
                    return `
                        <div class="signal-item" data-signal="${name}">
                            <div class="signal-header">
                                <span class="signal-name">${this.formatSignalName(name)}</span>
                                <span class="signal-score" style="color: ${color}">${score.toFixed(2)}</span>
                            </div>
                            <div class="signal-bar">
                                <div class="signal-fill" style="width: ${score * 100}%; background: ${color}"></div>
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    }

    /**
     * Render key findings
     */
    renderKeyFindings(results, mode) {
        const findings = results.key_findings || results.findings || [];
        
        if (findings.length === 0) {
            // Generate findings from interpretation
            if (results.interpretation) {
                return `
                    <div class="key-findings">
                        <h4> Analysis</h4>
                        <p class="finding-text">${results.interpretation}</p>
                    </div>
                `;
            }
            return '';
        }
        
        return `
            <div class="key-findings">
                <h4> Key Findings</h4>
                ${findings.map((finding, i) => `
                    <div class="finding-item">
                        <span class="finding-number">${i + 1}</span>
                        <span class="finding-text">${typeof finding === 'string' ? finding : finding.description || finding.text}</span>
                    </div>
                `).join('')}
            </div>
        `;
    }

    /**
     * Render affected claims
     */
    renderAffectedClaims(results, mode) {
        const claims = results.affected_claims || results.suppressed_claims || 
                      results.coordinated_claims || results.anomalous_claims || [];
        
        if (claims.length === 0) return '';
        
        // Handle both array of IDs and array of objects
        const claimList = typeof claims[0] === 'string' 
            ? claims.map(id => ({ id })) 
            : claims;
        
        return `
            <div class="affected-claims">
                <h4>📋 Affected Claims (${claimList.length})</h4>
                <div class="claims-list">
                    ${claimList.slice(0, 50).map((claim, index) => `
                        <div class="claim-item" data-id="${claim.id || claim.claim_id}">
                            <span class="claim-number">${index + 1}.</span>
                            <div class="claim-content">
                                <div class="claim-text">${this.truncate(claim.claim_text || claim.text || claim.id, 120)}</div>
                                ${claim.suppression_score ? `
                                    <div class="claim-score" style="color: ${getScoreColor(claim.suppression_score)}">
                                        ${(claim.suppression_score * 100).toFixed(0)}% suppression
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                    `).join('')}
                </div>
                ${claimList.length > 50 ? `
                    <div class="claims-more">
                        And ${claimList.length - 50} more...
                    </div>
                ` : ''}
            </div>
        `;
    }

    /**
     * Attach result interaction handlers
     */
    attachResultHandlers() {
        // Export button
        const exportBtn = this.container.querySelector('#export-results-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportResults());
        }
        
        // Try another mode button
        const tryAnotherBtn = this.container.querySelector('#try-another-btn');
        if (tryAnotherBtn) {
            tryAnotherBtn.addEventListener('click', () => this.showModeSelector());
        }
        
        // Signal expansion
        this.container.querySelectorAll('.signal-item').forEach(item => {
            item.addEventListener('click', () => this.expandSignal(item.dataset.signal));
        });
        
        // Claim click
        this.container.querySelectorAll('.claim-item, .result-item').forEach(item => {
            item.addEventListener('click', () => this.focusClaim(item.dataset.id));
        });
    }

    /**
     * Export results to file
     */
    exportResults() {
        if (!this.lastResults) return;
        
        const data = {
            mode: this.activeMode,
            query: this.searchInput.value,
            timestamp: new Date().toISOString(),
            claim_ids_count: this.lastClaimIds?.length || 0,
            results: this.lastResults
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `aegis-${this.activeMode}-${Date.now()}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
    }

    /**
     * Show mode selector dropdown
     */
    showModeSelector() {
        // Scroll to mode buttons
        this.container.querySelector('.detection-mode-selector').scrollIntoView({ 
            behavior: 'smooth' 
        });
    }

    /**
     * Expand signal details (placeholder)
     */
    expandSignal(signalName) {
        console.log('Expand signal:', signalName);
        // TODO: Show detailed signal analysis
    }

    /**
     * Focus claim in graph (placeholder)
     */
    focusClaim(claimId) {
        console.log('Focus claim:', claimId);
        // TODO: Pan graph to claim, highlight it
    }

    /**
     * Display error message
     */
    displayError(message) {
        const resultsPanel = this.container.querySelector('#detection-results');
        resultsPanel.innerHTML = `
            <div class="detection-error">
                <span class="error-icon">⚠️</span>
                <span class="error-text">${message}</span>
                <button class="btn-small retry-btn">Retry</button>
            </div>
        `;
        resultsPanel.style.display = 'block';
        
        resultsPanel.querySelector('.retry-btn').addEventListener('click', () => {
            this.executeSearch();
        });
    }

    /**
     * Show loading state
     */
    showLoading(mainText = 'Analyzing patterns...', subText = 'This may take 15-30 seconds') {
        this.isLoading = true;
        this.searchButton.disabled = true;
        this.updateLoadingText(mainText, subText);
        this.container.querySelector('#detection-loading').style.display = 'flex';
        this.hideResults();
    }

    /**
     * Hide loading state
     */
    hideLoading() {
        this.isLoading = false;
        this.searchButton.disabled = false;
        this.container.querySelector('#detection-loading').style.display = 'none';
    }

    /**
     * Hide results panel
     */
    showModePlaceholder(mode) {
        const resultsPanel = this.container.querySelector('#detection-results');
        const config = CONFIG.DETECTION_MODES[mode];
        
        resultsPanel.innerHTML = `
            <div class="mode-placeholder" style="text-align: center; padding: 60px 20px; color: #888;">
                <div style="font-size: 48px; margin-bottom: 20px; opacity: 0.5;">${config.icon}</div>
                <h3 style="margin: 0 0 10px 0; color: #aaa;">${config.name}</h3>
                <p style="margin: 0 0 20px 0;">Enter a search term and click <strong>${config.buttonText}</strong> to analyze.</p>
                <p style="font-size: 12px; color: #666;">${this.getExamplesForMode(mode)}</p>
            </div>
        `;
        resultsPanel.style.display = 'block';
    }
    
    hideResults() {
        this.container.querySelector('#detection-results').style.display = 'none';
    }

    /**
     * Format signal name for display
     */
    formatSignalName(name) {
        return name
            .replace(/_/g, ' ')
            .replace(/\b\w/g, c => c.toUpperCase());
    }

    /**
     * Truncate text
     */
    truncate(text, maxLength) {
        if (!text) return '';
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength - 3) + '...';
    }

    /**
     * Get current mode
     */
    getActiveMode() {
        return this.activeMode;
    }

    /**
     * Get last results
     */
    getLastResults() {
        return this.lastResults;
    }
    
    /**
     * Get claim IDs from last pattern search (for debugging)
     */
    getLastClaimIds() {
        return this.lastClaimIds;
    }
}
