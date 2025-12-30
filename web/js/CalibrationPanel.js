/**
 * Aegis Insight - Calibration Panel Component
 * 
 * Provides UI for:
 * - Profile selection dropdown
 * - Profile listing and management
 * - Weight sliders with normalization
 * - Pattern editing (include/exclude lists)
 * - Goldfinger threshold configuration
 * - Save/Load/Clone/Delete operations
 * 
 * Add to Admin panel as a new tab section.
 * 
 * Author: Aegis Insight Team
 * Date: November 2025
 */

class CalibrationPanel {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.profiles = [];
        this.currentProfile = null;
        this.isDirty = false;
        
        // API base URL
        this.apiBase = '/api';
        
        this.init();
    }
    
    async init() {
        this.render();
        await this.loadProfiles();
    }
    
    render() {
        this.container.innerHTML = `
            <div class="calibration-panel">
                <div class="calibration-header">
                    <h3>🎯 Calibration Profiles</h3>
                    <p class="calibration-subtitle">
                        Tune detection for your specific corpus and use case
                    </p>
                </div>
                
                <div class="calibration-content">
                    <!-- Profile Selector -->
                    <div class="profile-selector-section">
                        <label>Active Profile:</label>
                        <div class="profile-selector-row">
                            <select id="profile-select" class="profile-select">
                                <option value="">Loading profiles...</option>
                            </select>
                            <button id="btn-refresh-profiles" class="btn-icon" title="Refresh">🔄</button>
                            <button id="btn-new-profile" class="btn-icon" title="New Profile">➕</button>
                        </div>
                    </div>
                    
                    <!-- Profile Details -->
                    <div id="profile-details" class="profile-details hidden">
                        <!-- Metadata -->
                        <div class="profile-section">
                            <h4>Profile Info</h4>
                            <div class="form-group">
                                <label>Name:</label>
                                <input type="text" id="profile-name" class="form-input" />
                            </div>
                            <div class="form-group">
                                <label>Description:</label>
                                <textarea id="profile-description" class="form-textarea" rows="2"></textarea>
                            </div>
                            <div class="form-group">
                                <label>Tags:</label>
                                <input type="text" id="profile-tags" class="form-input" 
                                       placeholder="comma-separated tags" />
                            </div>
                        </div>
                        
                        <!-- Goldfinger Scoring -->
                        <div class="profile-section">
                            <h4>🎯 Goldfinger Scoring</h4>
                            <p class="section-help">
                                "Once is happenstance, twice is coincidence, three times is enemy action"
                            </p>
                            
                            <div class="scoring-grid">
                                <div class="scoring-item">
                                    <label>Happenstance Max:</label>
                                    <input type="number" id="scoring-happenstance" 
                                           class="form-input-small" min="1" max="5" value="1" />
                                    <span class="help-text">Indicators before "coincidence"</span>
                                </div>
                                <div class="scoring-item">
                                    <label>Coincidence Max:</label>
                                    <input type="number" id="scoring-coincidence" 
                                           class="form-input-small" min="2" max="10" value="2" />
                                    <span class="help-text">Indicators before "enemy action"</span>
                                </div>
                                <div class="scoring-item">
                                    <label>Enemy Action Base:</label>
                                    <input type="number" id="scoring-base" 
                                           class="form-input-small" min="0.3" max="0.8" step="0.05" value="0.50" />
                                    <span class="help-text">Base score at threshold</span>
                                </div>
                                <div class="scoring-item">
                                    <label>Log Factor:</label>
                                    <input type="number" id="scoring-factor" 
                                           class="form-input-small" min="0.1" max="0.4" step="0.02" value="0.20" />
                                    <span class="help-text">Growth rate after threshold</span>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Signal Weights -->
                        <div class="profile-section">
                            <h4>⚖️ Signal Weights</h4>
                            <p class="section-help">
                                Adjust importance of each signal. Weights auto-normalize to 1.0.
                            </p>
                            
                            <div id="weight-sliders" class="weight-sliders">
                                <!-- Populated dynamically -->
                            </div>
                            
                            <div class="weight-total">
                                Total: <span id="weight-total">1.00</span>
                                <span id="weight-status" class="weight-status">✓</span>
                            </div>
                        </div>
                        
                        <!-- Semantic Patterns -->
                        <div class="profile-section">
                            <h4>🔍 Semantic Patterns</h4>
                            <p class="section-help">
                                Define patterns to detect in PRIMARY claims. Uses embedding similarity.
                            </p>
                            
                            <div class="pattern-config">
                                <div class="form-group-inline">
                                    <label>Match Mode:</label>
                                    <select id="pattern-match-mode" class="form-select-small">
                                        <option value="semantic">Semantic (0.75)</option>
                                        <option value="broad">Broad (0.60)</option>
                                        <option value="exact">Exact (0.95)</option>
                                    </select>
                                </div>
                                <div class="form-group-inline">
                                    <label>Threshold:</label>
                                    <input type="number" id="pattern-threshold" 
                                           class="form-input-small" min="0.5" max="0.99" step="0.05" value="0.75" />
                                </div>
                            </div>
                            
                            <div class="pattern-categories">
                                <div class="pattern-category">
                                    <h5>Suppression Experiences</h5>
                                    <span class="category-help">Personal consequences: imprisoned, vilified, poverty</span>
                                    <div class="pattern-lists">
                                        <div class="pattern-list">
                                            <label>Include:</label>
                                            <textarea id="patterns-experiences-include" 
                                                      class="pattern-textarea" rows="3"
                                                      placeholder="one pattern per line"></textarea>
                                        </div>
                                        <div class="pattern-list">
                                            <label>Exclude:</label>
                                            <textarea id="patterns-experiences-exclude" 
                                                      class="pattern-textarea" rows="2"
                                                      placeholder="exceptions"></textarea>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="pattern-category">
                                    <h5>Institutional Actions</h5>
                                    <span class="category-help">Formal actions: charged with, fired, deplatformed</span>
                                    <div class="pattern-lists">
                                        <div class="pattern-list">
                                            <label>Include:</label>
                                            <textarea id="patterns-actions-include" 
                                                      class="pattern-textarea" rows="3"></textarea>
                                        </div>
                                        <div class="pattern-list">
                                            <label>Exclude:</label>
                                            <textarea id="patterns-actions-exclude" 
                                                      class="pattern-textarea" rows="2"></textarea>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="pattern-category">
                                    <h5>Dismissal Language</h5>
                                    <span class="category-help">Delegitimizing: debunked, conspiracy theory, seditious</span>
                                    <div class="pattern-lists">
                                        <div class="pattern-list">
                                            <label>Include:</label>
                                            <textarea id="patterns-dismissal-include" 
                                                      class="pattern-textarea" rows="3"></textarea>
                                        </div>
                                        <div class="pattern-list">
                                            <label>Exclude:</label>
                                            <textarea id="patterns-dismissal-exclude" 
                                                      class="pattern-textarea" rows="2"></textarea>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="pattern-category">
                                    <h5>Suppression of Record</h5>
                                    <span class="category-help">Historical erasure: omitted from textbooks, forgotten</span>
                                    <div class="pattern-lists">
                                        <div class="pattern-list">
                                            <label>Include:</label>
                                            <textarea id="patterns-record-include" 
                                                      class="pattern-textarea" rows="3"></textarea>
                                        </div>
                                        <div class="pattern-list">
                                            <label>Exclude:</label>
                                            <textarea id="patterns-record-exclude" 
                                                      class="pattern-textarea" rows="2"></textarea>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Actions -->
                        <div class="profile-actions">
                            <button id="btn-save-profile" class="btn btn-primary">💾 Save</button>
                            <button id="btn-clone-profile" class="btn btn-secondary">📋 Clone</button>
                            <button id="btn-reset-profile" class="btn btn-secondary">↩️ Reset</button>
                            <button id="btn-delete-profile" class="btn btn-danger">🗑️ Delete</button>
                            <button id="btn-test-profile" class="btn btn-accent">🧪 Test</button>
                        </div>
                    </div>
                    
                    <!-- Test Panel -->
                    <div id="test-panel" class="test-panel hidden">
                        <h4>🧪 Test Profile</h4>
                        <div class="test-input-row">
                            <input type="text" id="test-topic" class="form-input" 
                                   placeholder="Enter topic to test (e.g., Thomas Paine)" />
                            <button id="btn-run-test" class="btn btn-primary">Run Detection</button>
                        </div>
                        <div id="test-results" class="test-results">
                            <!-- Results appear here -->
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        this.bindEvents();
        this.renderWeightSliders();
    }
    
    bindEvents() {
        // Profile selector
        document.getElementById('profile-select').addEventListener('change', (e) => {
            this.selectProfile(e.target.value);
        });
        
        document.getElementById('btn-refresh-profiles').addEventListener('click', () => {
            this.loadProfiles();
        });
        
        document.getElementById('btn-new-profile').addEventListener('click', () => {
            this.createNewProfile();
        });
        
        // Actions
        document.getElementById('btn-save-profile').addEventListener('click', () => {
            this.saveProfile();
        });
        
        document.getElementById('btn-clone-profile').addEventListener('click', () => {
            this.cloneProfile();
        });
        
        document.getElementById('btn-reset-profile').addEventListener('click', () => {
            this.resetProfile();
        });
        
        document.getElementById('btn-delete-profile').addEventListener('click', () => {
            this.deleteProfile();
        });
        
        document.getElementById('btn-test-profile').addEventListener('click', () => {
            this.toggleTestPanel();
        });
        
        document.getElementById('btn-run-test').addEventListener('click', () => {
            this.runTest();
        });
        
        // Mark dirty on changes
        this.container.querySelectorAll('input, textarea, select').forEach(el => {
            el.addEventListener('change', () => {
                this.isDirty = true;
            });
        });
    }
    
    renderWeightSliders() {
        const container = document.getElementById('weight-sliders');
        
        const signals = [
            { id: 'suppression_narrative', label: 'Suppression Narrative', default: 0.40 },
            { id: 'meta_claim_density', label: 'META Claim Density', default: 0.15 },
            { id: 'network_isolation', label: 'Network Isolation', default: 0.10 },
            { id: 'evidence_avoidance', label: 'Evidence Avoidance', default: 0.20 },
            { id: 'authority_mismatch', label: 'Authority Mismatch', default: 0.15 }
        ];
        
        container.innerHTML = signals.map(signal => `
            <div class="weight-slider-row">
                <label>${signal.label}:</label>
                <input type="range" 
                       id="weight-${signal.id}" 
                       class="weight-slider"
                       min="0" max="100" value="${signal.default * 100}"
                       data-signal="${signal.id}" />
                <span class="weight-value" id="weight-value-${signal.id}">${signal.default.toFixed(2)}</span>
            </div>
        `).join('');
        
        // Bind slider events
        container.querySelectorAll('.weight-slider').forEach(slider => {
            slider.addEventListener('input', (e) => {
                this.updateWeightDisplay(e.target.dataset.signal, e.target.value);
                this.updateWeightTotal();
                this.isDirty = true;
            });
        });
    }
    
    updateWeightDisplay(signal, value) {
        const displayValue = (value / 100).toFixed(2);
        document.getElementById(`weight-value-${signal}`).textContent = displayValue;
    }
    
    updateWeightTotal() {
        let total = 0;
        document.querySelectorAll('.weight-slider').forEach(slider => {
            total += parseInt(slider.value) / 100;
        });
        
        const totalEl = document.getElementById('weight-total');
        const statusEl = document.getElementById('weight-status');
        
        totalEl.textContent = total.toFixed(2);
        
        if (Math.abs(total - 1.0) < 0.01) {
            statusEl.textContent = '✓';
            statusEl.className = 'weight-status valid';
        } else {
            statusEl.textContent = '(will normalize)';
            statusEl.className = 'weight-status warning';
        }
    }
    
    async loadProfiles() {
        try {
            const response = await fetch(`${this.apiBase}/profiles`);
            const data = await response.json();
            
            this.profiles = data.profiles || [];
            this.updateProfileSelect();
            
        } catch (error) {
            console.error('Failed to load profiles:', error);
            this.showError('Failed to load profiles');
        }
    }
    
    updateProfileSelect() {
        const select = document.getElementById('profile-select');
        
        select.innerHTML = this.profiles.map(p => `
            <option value="${p.filename}">${p.name}</option>
        `).join('');
        
        // Select first profile
        if (this.profiles.length > 0) {
            select.value = this.profiles[0].filename;
            this.selectProfile(this.profiles[0].filename);
        }
    }
    
    async selectProfile(filename) {
        if (!filename) return;
        
        if (this.isDirty) {
            if (!confirm('You have unsaved changes. Discard?')) {
                return;
            }
        }
        
        try {
            const response = await fetch(`${this.apiBase}/profiles/${filename}`);
            const data = await response.json();
            
            this.currentProfile = data.profile;
            this.currentFilename = filename;
            this.populateForm(data.profile);
            this.isDirty = false;
            
            document.getElementById('profile-details').classList.remove('hidden');
            
        } catch (error) {
            console.error('Failed to load profile:', error);
            this.showError('Failed to load profile');
        }
    }
    
    populateForm(profile) {
        // Metadata
        document.getElementById('profile-name').value = profile.metadata?.name || '';
        document.getElementById('profile-description').value = profile.metadata?.description || '';
        document.getElementById('profile-tags').value = (profile.metadata?.tags || []).join(', ');
        
        // Scoring
        const scoring = profile.scoring || {};
        document.getElementById('scoring-happenstance').value = scoring.happenstance_max || 1;
        document.getElementById('scoring-coincidence').value = scoring.coincidence_max || 2;
        document.getElementById('scoring-base').value = scoring.enemy_action_base || 0.50;
        document.getElementById('scoring-factor').value = scoring.logarithmic_factor || 0.20;
        
        // Weights
        const weights = profile.signal_weights || {};
        Object.keys(weights).forEach(signal => {
            const slider = document.getElementById(`weight-${signal}`);
            if (slider) {
                slider.value = weights[signal] * 100;
                this.updateWeightDisplay(signal, slider.value);
            }
        });
        this.updateWeightTotal();
        
        // Patterns
        const patterns = profile.semantic_patterns || {};
        document.getElementById('pattern-match-mode').value = patterns.match_mode || 'semantic';
        document.getElementById('pattern-threshold').value = patterns.match_threshold || 0.75;
        
        // Pattern categories
        this.setPatternTextarea('patterns-experiences-include', 
            patterns.suppression_experiences?.include);
        this.setPatternTextarea('patterns-experiences-exclude', 
            patterns.suppression_experiences?.exclude);
        
        this.setPatternTextarea('patterns-actions-include', 
            patterns.institutional_actions?.include);
        this.setPatternTextarea('patterns-actions-exclude', 
            patterns.institutional_actions?.exclude);
        
        this.setPatternTextarea('patterns-dismissal-include', 
            patterns.dismissal_language?.include);
        this.setPatternTextarea('patterns-dismissal-exclude', 
            patterns.dismissal_language?.exclude);
        
        this.setPatternTextarea('patterns-record-include', 
            patterns.suppression_of_record?.include);
        this.setPatternTextarea('patterns-record-exclude', 
            patterns.suppression_of_record?.exclude);
    }
    
    setPatternTextarea(id, patterns) {
        const el = document.getElementById(id);
        if (el) {
            el.value = (patterns || []).join('\n');
        }
    }
    
    getPatternArray(id) {
        const el = document.getElementById(id);
        if (!el) return [];
        return el.value.split('\n')
            .map(s => s.trim())
            .filter(s => s.length > 0);
    }
    
    buildProfileFromForm() {
        // Collect weights
        const weights = {};
        document.querySelectorAll('.weight-slider').forEach(slider => {
            weights[slider.dataset.signal] = parseInt(slider.value) / 100;
        });
        
        return {
            schema_version: "2.0",
            metadata: {
                name: document.getElementById('profile-name').value,
                description: document.getElementById('profile-description').value,
                author: this.currentProfile?.metadata?.author || 'user',
                tags: document.getElementById('profile-tags').value
                    .split(',').map(s => s.trim()).filter(s => s)
            },
            scoring: {
                mode: "goldfinger",
                happenstance_max: parseInt(document.getElementById('scoring-happenstance').value),
                coincidence_max: parseInt(document.getElementById('scoring-coincidence').value),
                enemy_action_base: parseFloat(document.getElementById('scoring-base').value),
                logarithmic_factor: parseFloat(document.getElementById('scoring-factor').value),
                max_score: 0.95
            },
            semantic_patterns: {
                match_mode: document.getElementById('pattern-match-mode').value,
                match_threshold: parseFloat(document.getElementById('pattern-threshold').value),
                suppression_experiences: {
                    include: this.getPatternArray('patterns-experiences-include'),
                    exclude: this.getPatternArray('patterns-experiences-exclude')
                },
                institutional_actions: {
                    include: this.getPatternArray('patterns-actions-include'),
                    exclude: this.getPatternArray('patterns-actions-exclude')
                },
                dismissal_language: {
                    include: this.getPatternArray('patterns-dismissal-include'),
                    exclude: this.getPatternArray('patterns-dismissal-exclude')
                },
                suppression_of_record: {
                    include: this.getPatternArray('patterns-record-include'),
                    exclude: this.getPatternArray('patterns-record-exclude')
                }
            },
            signal_weights: weights,
            thresholds: this.currentProfile?.thresholds || {
                min_claims_for_analysis: 5,
                score_levels: {
                    critical: 0.80,
                    high: 0.60,
                    moderate: 0.40,
                    low: 0.20
                }
            }
        };
    }
    
    async saveProfile() {
        const profile = this.buildProfileFromForm();
        
        try {
            const response = await fetch(`${this.apiBase}/profiles/${this.currentFilename}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ profile })
            });
            
            if (!response.ok) {
                throw new Error('Save failed');
            }
            
            const data = await response.json();
            this.showSuccess('Profile saved successfully');
            this.isDirty = false;
            this.loadProfiles();
            
        } catch (error) {
            console.error('Failed to save profile:', error);
            this.showError('Failed to save profile');
        }
    }
    
    async cloneProfile() {
        const newName = prompt('Enter name for cloned profile:', 
            this.currentProfile?.metadata?.name + ' (copy)');
        
        if (!newName) return;
        
        try {
            const response = await fetch(
                `${this.apiBase}/profiles/${this.currentFilename}/clone?new_name=${encodeURIComponent(newName)}`,
                { method: 'POST' }
            );
            
            if (!response.ok) {
                throw new Error('Clone failed');
            }
            
            const data = await response.json();
            this.showSuccess(`Profile cloned as '${newName}'`);
            await this.loadProfiles();
            
            // Select the new profile
            document.getElementById('profile-select').value = data.filename;
            this.selectProfile(data.filename);
            
        } catch (error) {
            console.error('Failed to clone profile:', error);
            this.showError('Failed to clone profile');
        }
    }
    
    resetProfile() {
        if (this.currentFilename) {
            this.selectProfile(this.currentFilename);
        }
    }
    
    async deleteProfile() {
        if (!confirm(`Delete profile "${this.currentProfile?.metadata?.name}"? This cannot be undone.`)) {
            return;
        }
        
        try {
            const response = await fetch(`${this.apiBase}/profiles/${this.currentFilename}`, {
                method: 'DELETE'
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Delete failed');
            }
            
            this.showSuccess('Profile deleted');
            this.currentProfile = null;
            this.currentFilename = null;
            document.getElementById('profile-details').classList.add('hidden');
            await this.loadProfiles();
            
        } catch (error) {
            console.error('Failed to delete profile:', error);
            this.showError(error.message || 'Failed to delete profile');
        }
    }
    
    createNewProfile() {
        const name = prompt('Enter name for new profile:');
        if (!name) return;
        
        // Start with default profile
        this.currentProfile = {
            metadata: { name, description: '', tags: [] },
            scoring: {
                mode: 'goldfinger',
                happenstance_max: 1,
                coincidence_max: 2,
                enemy_action_base: 0.50,
                logarithmic_factor: 0.20,
                max_score: 0.95
            },
            semantic_patterns: {
                match_mode: 'semantic',
                match_threshold: 0.75,
                suppression_experiences: { include: [], exclude: [] },
                institutional_actions: { include: [], exclude: [] },
                dismissal_language: { include: [], exclude: [] },
                suppression_of_record: { include: [], exclude: [] }
            },
            signal_weights: {
                suppression_narrative: 0.40,
                meta_claim_density: 0.15,
                network_isolation: 0.10,
                evidence_avoidance: 0.20,
                authority_mismatch: 0.15
            },
            thresholds: {
                min_claims_for_analysis: 5,
                score_levels: { critical: 0.80, high: 0.60, moderate: 0.40, low: 0.20 }
            }
        };
        
        this.currentFilename = name.toLowerCase().replace(/\s+/g, '_') + '.json';
        this.populateForm(this.currentProfile);
        document.getElementById('profile-details').classList.remove('hidden');
        this.isDirty = true;
        
        // Save immediately as new
        this.saveAsNew();
    }
    
    async saveAsNew() {
        const profile = this.buildProfileFromForm();
        
        try {
            const response = await fetch(`${this.apiBase}/profiles`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ profile, filename: this.currentFilename })
            });
            
            if (!response.ok) {
                throw new Error('Create failed');
            }
            
            const data = await response.json();
            this.showSuccess('Profile created successfully');
            this.isDirty = false;
            await this.loadProfiles();
            
            document.getElementById('profile-select').value = data.filename;
            
        } catch (error) {
            console.error('Failed to create profile:', error);
            this.showError('Failed to create profile');
        }
    }
    
    toggleTestPanel() {
        const panel = document.getElementById('test-panel');
        panel.classList.toggle('hidden');
    }
    
    async runTest() {
        const topic = document.getElementById('test-topic').value.trim();
        if (!topic) {
            this.showError('Please enter a topic to test');
            return;
        }
        
        const resultsDiv = document.getElementById('test-results');
        resultsDiv.innerHTML = '<div class="loading">Running detection...</div>';
        
        // Build current profile from form (may have unsaved changes)
        const profile = this.buildProfileFromForm();
        
        try {
            const response = await fetch(`${this.apiBase}/detect/suppression/with-profile?topic=${encodeURIComponent(topic)}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(profile)
            });
            
            const data = await response.json();
            this.displayTestResults(data.result);
            
        } catch (error) {
            console.error('Test failed:', error);
            resultsDiv.innerHTML = `<div class="error">Test failed: ${error.message}</div>`;
        }
    }
    
    displayTestResults(result) {
        const resultsDiv = document.getElementById('test-results');
        
        const scoreClass = result.level.toLowerCase();
        const narrativeSignal = result.signals?.suppression_narrative || {};
        
        resultsDiv.innerHTML = `
            <div class="test-result ${scoreClass}">
                <div class="test-score">
                    <span class="score-value">${result.suppression_score.toFixed(3)}</span>
                    <span class="score-level">${result.level}</span>
                </div>
                
                <div class="test-meta">
                    <span>Claims analyzed: ${result.claims_analyzed}</span>
                    <span>Confidence: ${(result.confidence * 100).toFixed(0)}%</span>
                    <span>Profile: ${result.profile_used}</span>
                </div>
                
                <div class="test-interpretation">
                    ${result.interpretation}
                </div>
                
                <div class="test-narrative">
                    <h5>Suppression Narrative Signal</h5>
                    <p>Indicators found: <strong>${narrativeSignal.indicators_found || 0}</strong></p>
                    ${this.formatCategoryBreakdown(narrativeSignal.category_breakdown)}
                </div>
                
                ${this.formatIndicatorDetails(narrativeSignal.details)}
                
                <div class="test-signals">
                    <h5>All Signals</h5>
                    ${this.formatSignals(result.signals)}
                </div>
            </div>
        `;
    }
    
    formatCategoryBreakdown(breakdown) {
        if (!breakdown || Object.keys(breakdown).length === 0) {
            return '<p class="no-data">No indicators detected</p>';
        }
        
        return `
            <div class="category-breakdown">
                ${Object.entries(breakdown).map(([cat, count]) => `
                    <span class="category-badge">${cat.replace(/_/g, ' ')}: ${count}</span>
                `).join('')}
            </div>
        `;
    }
    
    formatIndicatorDetails(details) {
        if (!details || details.length === 0) return '';
        
        return `
            <div class="indicator-details">
                <h5>Detected Indicators (top ${Math.min(details.length, 10)})</h5>
                <ul>
                    ${details.slice(0, 10).map(ind => `
                        <li>
                            <span class="indicator-category">[${ind.category}]</span>
                            <span class="indicator-pattern">matched: "${ind.matched_pattern}"</span>
                            <span class="indicator-similarity">(${(ind.similarity * 100).toFixed(0)}%)</span>
                            <div class="indicator-claim">${ind.claim_text.substring(0, 150)}...</div>
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;
    }
    
    formatSignals(signals) {
        if (!signals) return '';
        
        return `
            <table class="signals-table">
                <thead>
                    <tr>
                        <th>Signal</th>
                        <th>Score</th>
                    </tr>
                </thead>
                <tbody>
                    ${Object.entries(signals).map(([name, data]) => `
                        <tr>
                            <td>${name.replace(/_/g, ' ')}</td>
                            <td>${(data.score || 0).toFixed(3)}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    }
    
    showSuccess(message) {
        // You can integrate with your existing notification system
        console.log('Success:', message);
        alert(message);
    }
    
    showError(message) {
        console.error('Error:', message);
        alert('Error: ' + message);
    }
}


// === Profile Selector for Detection Tab ===

class ProfileSelector {
    /**
     * Simple profile selector dropdown for the Detection tab
     * 
     * Usage:
     *   const selector = new ProfileSelector('detection-profile-selector');
     *   const profile = await selector.getSelectedProfile();
     */
    
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.apiBase = '/api';
        this.profiles = [];
        this.selectedFilename = 'default.json';
        
        this.init();
    }
    
    async init() {
        this.render();
        await this.loadProfiles();
    }
    
    render() {
        this.container.innerHTML = `
            <div class="profile-selector-compact">
                <label>Profile:</label>
                <select id="detection-profile-select" class="profile-select-compact">
                    <option value="default.json">Loading...</option>
                </select>
                <span id="profile-description-tooltip" class="profile-tooltip"></span>
            </div>
        `;
        
        document.getElementById('detection-profile-select').addEventListener('change', (e) => {
            this.selectedFilename = e.target.value;
            this.updateTooltip();
        });
    }
    
    async loadProfiles() {
        try {
            const response = await fetch(`${this.apiBase}/profiles`);
            const data = await response.json();
            
            this.profiles = data.profiles || [];
            
            const select = document.getElementById('detection-profile-select');
            select.innerHTML = this.profiles.map(p => `
                <option value="${p.filename}" 
                        data-description="${p.description || ''}"
                        ${p.filename === 'default.json' ? 'selected' : ''}>
                    ${p.name}
                </option>
            `).join('');
            
            this.updateTooltip();
            
        } catch (error) {
            console.error('Failed to load profiles:', error);
        }
    }
    
    updateTooltip() {
        const select = document.getElementById('detection-profile-select');
        const tooltip = document.getElementById('profile-description-tooltip');
        const selected = select.options[select.selectedIndex];
        
        if (selected && tooltip) {
            tooltip.textContent = selected.dataset.description || '';
        }
    }
    
    getSelectedFilename() {
        return this.selectedFilename;
    }
    
    async getSelectedProfile() {
        try {
            const response = await fetch(`${this.apiBase}/profiles/${this.selectedFilename}`);
            const data = await response.json();
            return data.profile;
        } catch (error) {
            console.error('Failed to load selected profile:', error);
            return null;
        }
    }
}


// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CalibrationPanel, ProfileSelector };
}
