/**
 * Data Loading Wizard
 * 
 * Features:
 * - Browse files in inbox
 * - Select specific files to import
 * - Real-time progress tracking
 * - Live console output
 * - Stats display
 */

export class DataWizard {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.files = [];
        this.selectedFiles = [];
        this.currentJobId = null;
        this.pollInterval = null;
        this.inboxPath = '';
        
        if (!this.container) {
            console.error('DataWizard: Container not found:', containerId);
            return;
        }
        
        this.init();
    }
    
    async init() {
        this.renderLoading();
        await this.loadFiles();
        this.renderFileList();
    }
    
    renderLoading() {
        this.container.innerHTML = `
            <div class="wizard-loading">
                <div class="spinner"></div>
                <p>Loading inbox files...</p>
            </div>
        `;
    }
    
    async loadFiles() {
        try {
            const response = await fetch('/api/list-inbox-files');
            const data = await response.json();
            
            if (data.success) {
                this.files = data.files;
                this.inboxPath = data.inbox_path;
            } else {
                console.error('Failed to load files:', data.error);
                this.files = [];
            }
        } catch (error) {
            console.error('Error loading files:', error);
            this.files = [];
        }
    }
    
    renderFileList() {
        this.container.innerHTML = `
            <div class="data-wizard">
                <div class="wizard-header">
                    <h2> Data Import</h2>
                    <p class="inbox-path">Inbox: <code>${this.escapeHtml(this.inboxPath)}</code></p>
                </div>
                
                ${this.files.length > 0 ? `
                    <div class="file-list-container">
                        <div class="file-list-header">
                            <label class="select-all-label">
                                <input type="checkbox" id="select-all-files">
                                <span>Select All (${this.files.length} files)</span>
                            </label>
                            <div class="file-stats">
                                <span id="selected-count">0 selected</span>
                                <span id="selected-size">0 MB</span>
                            </div>
                        </div>
                        
                        <div class="file-list" id="file-list">
                            ${this.renderFiles()}
                        </div>
                    </div>
                    
                    <div class="wizard-actions">
                        <button id="refresh-files" class="btn btn-secondary">
                             Refresh
                        </button>
                        <button id="start-import" class="btn btn-primary" disabled>
                             Import Selected Files
                        </button>
                    </div>
                ` : `
                    <div class="empty-state">
                        <div class="empty-icon"></div>
                        <p class="empty-title">No PDF files found</p>
                        <p class="empty-hint">Place PDF files in:<br><code>${this.escapeHtml(this.inboxPath)}</code></p>
                        <button id="refresh-files" class="btn btn-primary">
                            ðŸ”„ Refresh
                        </button>
                    </div>
                `}
            </div>
        `;
        
        this.attachEventHandlers();
    }
    
    renderFiles() {
        return this.files.map((file, idx) => `
            <label class="file-item">
                <input type="checkbox" class="file-checkbox" data-index="${idx}">
                <div class="file-info">
                    <div class="file-name"> ${this.escapeHtml(file.filename)}</div>
                    <div class="file-size">${file.size_mb} MB</div>
                </div>
            </label>
        `).join('');
    }
    
    attachEventHandlers() {
        // Select all toggle
        const selectAll = document.getElementById('select-all-files');
        if (selectAll) {
            selectAll.addEventListener('change', (e) => {
                document.querySelectorAll('.file-checkbox').forEach(cb => {
                    cb.checked = e.target.checked;
                });
                this.updateSelection();
            });
        }
        
        // Individual file selection
        document.querySelectorAll('.file-checkbox').forEach(cb => {
            cb.addEventListener('change', () => this.updateSelection());
        });
        
        // Refresh button
        const refreshBtn = document.getElementById('refresh-files');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', async () => {
                this.init();
            });
        }
        
        // Start import button
        const startBtn = document.getElementById('start-import');
        if (startBtn) {
            startBtn.addEventListener('click', () => {
                this.startImport();
            });
        }
    }
    
    updateSelection() {
        const checkboxes = document.querySelectorAll('.file-checkbox:checked');
        this.selectedFiles = Array.from(checkboxes).map(cb => {
            const idx = parseInt(cb.dataset.index);
            return this.files[idx];
        });
        
        // Update UI
        const count = this.selectedFiles.length;
        const totalSize = this.selectedFiles.reduce((sum, f) => sum + f.size_mb, 0);
        
        const countEl = document.getElementById('selected-count');
        const sizeEl = document.getElementById('selected-size');
        const startBtn = document.getElementById('start-import');
        
        if (countEl) countEl.textContent = `${count} selected`;
        if (sizeEl) sizeEl.textContent = `${totalSize.toFixed(1)} MB`;
        if (startBtn) startBtn.disabled = count === 0;
    }
    
    async startImport() {
        const filenames = this.selectedFiles.map(f => f.filename);
        
        if (filenames.length === 0) {
            alert('Please select at least one file');
            return;
        }
        
        try {
            const response = await fetch('/api/load-pipeline', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    type: 'pdfs',
                    path: this.inboxPath,
                    selected_files: filenames
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.currentJobId = data.job_id;
                this.renderProgress();
                this.startPolling();
            } else {
                alert(`Error: ${data.error || 'Failed to start import'}`);
            }
        } catch (error) {
            console.error('Error starting import:', error);
            alert(`Error starting import: ${error.message}`);
        }
    }
    
    renderProgress() {
        this.container.innerHTML = `
            <div class="import-progress">
                <div class="progress-header">
                    <h2> Import in Progress...</h2>
                    <p class="job-id">Job ID: <code>${this.currentJobId}</code></p>
                </div>
                
                <div class="progress-bar-container">
                    <div class="progress-bar" id="progress-bar" style="width: 0%">
                        <span class="progress-text" id="progress-text">0%</span>
                    </div>
                </div>
                
                <div class="progress-stats">
                    <div class="stat-box">
                        <div class="stat-label">Status</div>
                        <div class="stat-value" id="job-status">Running</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Duration</div>
                        <div class="stat-value" id="job-duration">0s</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Documents</div>
                        <div class="stat-value" id="docs-count">---</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Entities</div>
                        <div class="stat-value" id="entities-count">---</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Claims</div>
                        <div class="stat-value" id="claims-count">---</div>
                    </div>
                </div>
                
                <div class="output-console">
                    <div class="console-header">
                        <h3>Live Output</h3>
                        <button id="toggle-console" class="btn-icon"></button>
                    </div>
                    <div class="console-content" id="console-output">
                        <p class="console-line">Starting import...</p>
                    </div>
                </div>
                
                <div class="progress-actions">
                    <button id="view-full-output" class="btn btn-secondary">
                         View Full Output
                    </button>
                    <button id="cancel-import" class="btn btn-secondary">
                         Cancel
                    </button>
                </div>
            </div>
        `;
        
        // Attach event handlers
        document.getElementById('toggle-console')?.addEventListener('click', (e) => {
            const console = document.getElementById('console-output');
            const button = e.target;
            if (console.style.display === 'none') {
                console.style.display = 'block';
                button.textContent = '';
            } else {
                console.style.display = 'none';
                button.textContent = '';
            }
        });
        
        document.getElementById('view-full-output')?.addEventListener('click', () => {
            this.showFullOutput();
        });
        
        document.getElementById('cancel-import')?.addEventListener('click', () => {
            if (confirm('Cancel this import?')) {
                this.stopPolling();
                this.init();
            }
        });
    }
    
    startPolling() {
        this.pollInterval = setInterval(() => {
            this.updateProgress();
        }, 2000);  // Poll every 2 seconds
    }
    
    async updateProgress() {
        if (!this.currentJobId) return;
        
        try {
            const response = await fetch(`/api/load-status/${this.currentJobId}`);
            const data = await response.json();
            
            if (!data.success) {
                console.error('Status fetch failed:', data.error);
                return;
            }
            
            // Update status
            const statusEl = document.getElementById('job-status');
            if (statusEl) {
                statusEl.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
                statusEl.className = `stat-value status-${data.status}`;
            }
            
            // Update duration
            if (data.duration !== undefined) {
                const durationEl = document.getElementById('job-duration');
                if (durationEl) {
                    const mins = Math.floor(data.duration / 60);
                    const secs = data.duration % 60;
                    durationEl.textContent = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
                }
            }
            
            // Update progress bar
            if (data.progress_percent !== undefined) {
                const pct = Math.min(data.progress_percent, 100);
                const bar = document.getElementById('progress-bar');
                const text = document.getElementById('progress-text');
                if (bar) bar.style.width = `${pct}%`;
                if (text) text.textContent = `${pct}%`;
            }
            
            // Update stats
            if (data.stats) {
                const docsEl = document.getElementById('docs-count');
                const entitiesEl = document.getElementById('entities-count');
                const claimsEl = document.getElementById('claims-count');
                
                if (docsEl) docsEl.textContent = data.stats.documents || 0;
                if (entitiesEl) entitiesEl.textContent = data.stats.entities || 0;
                if (claimsEl) claimsEl.textContent = data.stats.claims || 0;
            }
            
            // Update console output (last 10 lines, filtered)
            if (data.output && data.output.length > 0) {
                const consoleDiv = document.getElementById('console-output');
                if (consoleDiv) {
                    // Filter out noise: JSON parse errors (they're retried), progress bars, etc.
                    const filteredLines = data.output.filter(line => {
                        const lower = line.toLowerCase();
                        // Skip JSON parse errors (normal LLM chattiness, handled by retry logic)
                        if (lower.includes('json parse error') || lower.includes('extra data')) return false;
                        // Skip raw progress bar output
                        if (line.includes('|██') || line.includes('100%|')) return false;
                        // Skip empty lines
                        if (!line.trim()) return false;
                        return true;
                    });
                    
                    const lines = filteredLines.slice(-10).map(line => 
                        `<p class="console-line">${this.escapeHtml(line)}</p>`
                    ).join('');
                    consoleDiv.innerHTML = lines;
                    consoleDiv.scrollTop = consoleDiv.scrollHeight;
                }
            }
            
            // Check if complete
            if (data.status === 'completed') {
                this.stopPolling();
                this.renderComplete(data);
            } else if (data.status === 'error') {
                this.stopPolling();
                this.renderError(data);
            }
            
        } catch (error) {
            console.error('Error fetching status:', error);
        }
    }
    
    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }
    
    async showFullOutput() {
        try {
            const response = await fetch(`/api/load-output/${this.currentJobId}?lines=500`);
            const data = await response.json();
            
            if (data.success && data.output) {
                // Create modal overlay
                const modal = document.createElement('div');
                modal.className = 'output-modal';
                modal.innerHTML = `
                    <div class="modal-backdrop"></div>
                    <div class="modal-content">
                        <div class="modal-header">
                            <h3>Full Output (${data.total_lines} lines)</h3>
                            <button class="modal-close"></button>
                        </div>
                        <pre class="modal-output">${this.escapeHtml(data.output.join('\n'))}</pre>
                        <div class="modal-footer">
                            <button class="btn btn-secondary modal-close-btn">Close</button>
                        </div>
                    </div>
                `;
                
                document.body.appendChild(modal);
                
                // Close handlers
                modal.querySelector('.modal-close').addEventListener('click', () => {
                    document.body.removeChild(modal);
                });
                modal.querySelector('.modal-close-btn').addEventListener('click', () => {
                    document.body.removeChild(modal);
                });
                modal.querySelector('.modal-backdrop').addEventListener('click', () => {
                    document.body.removeChild(modal);
                });
            }
        } catch (error) {
            console.error('Error fetching full output:', error);
            alert('Failed to load full output');
        }
    }
    
    renderComplete(data) {
        this.container.innerHTML = `
            <div class="import-complete">
                <div class="completion-icon">&#10003;</div>
                <h2>Import Complete!</h2>
                
                ${data.stats ? `
                    <div class="completion-stats">
                        <div class="stat-card">
                            <div class="stat-value">${data.stats.documents || 0}</div>
                            <div class="stat-label">Documents</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${data.stats.entities || 0}</div>
                            <div class="stat-label">Entities</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${data.stats.claims || 0}</div>
                            <div class="stat-label">Claims</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${Math.floor((data.duration || 0) / 60)}m</div>
                            <div class="stat-label">Duration</div>
                        </div>
                    </div>
                ` : '<p>Import completed successfully</p>'}
                
                <div class="completion-actions">
                    <button id="import-more" class="btn btn-primary">
                         Import More Files
                    </button>
                    <button id="view-graph" class="btn btn-secondary">
                         View Knowledge Graph
                    </button>
                </div>
            </div>
        `;
        
        document.getElementById('import-more')?.addEventListener('click', () => {
            this.init();
        });
        
        document.getElementById('view-graph')?.addEventListener('click', () => {
            // Switch to Graph Search tab
            const graphTab = document.querySelector('[data-tab="graph-search"]');
            if (graphTab) graphTab.click();
        });
    }
    
    renderError(data) {
        this.container.innerHTML = `
            <div class="import-error">
                <div class="error-icon"></div>
                <h2>Import Failed</h2>
                <p class="error-message">${this.escapeHtml(data.error || 'Unknown error occurred')}</p>
                
                ${data.output && data.output.length > 0 ? `
                    <details class="error-details">
                        <summary>Show Output (last 50 lines)</summary>
                        <pre class="error-output">${this.escapeHtml(data.output.slice(-50).join('\n'))}</pre>
                    </details>
                ` : ''}
                
                <div class="error-actions">
                    <button id="try-again" class="btn btn-primary">
                         Try Again
                    </button>
                    <button id="view-logs" class="btn btn-secondary">
                         View Full Logs
                    </button>
                </div>
            </div>
        `;
        
        document.getElementById('try-again')?.addEventListener('click', () => {
            this.init();
        });
        
        document.getElementById('view-logs')?.addEventListener('click', () => {
            this.showFullOutput();
        });
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}
