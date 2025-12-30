/**
 * Data Management Component for Aegis Insight Admin Panel
 * 
 * Provides UI for:
 * - Viewing and managing source files
 * - Browsing, filtering, and searching claims
 * - Excluding/restoring claims from detection
 * - Deleting claims and sources
 * 
 * @author Aegis Development Team
 * @created 2025-12-07
 */

export class DataManagement {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container ${containerId} not found`);
            return;
        }
        
        this.apiBase = '';
        this.selectedSources = new Set();
        this.selectedClaims = new Set();
        this.currentSource = null;
        this.currentClaim = null;
        this.claimsData = [];
        this.sourcesData = [];
        
        // Pagination state
        this.claimsOffset = 0;
        this.claimsLimit = 50;
        this.claimsTotal = 0;
        
        // Filter state
        this.filters = {
            source: null,
            claim_type: null,
            status: 'active',
            search: ''
        };
        
        this.render();
        this.bindEvents();
        this.loadStats();
        this.loadSources();
    }
    
    render() {
        this.container.innerHTML = `
            <div class="data-management">
                <!-- Stats Overview -->
                <div class="dm-stats-bar" id="dm-stats-bar">
                    <div class="dm-stat">
                        <span class="dm-stat-value" id="stat-total-claims">-</span>
                        <span class="dm-stat-label">Total Claims</span>
                    </div>
                    <div class="dm-stat">
                        <span class="dm-stat-value" id="stat-active-claims">-</span>
                        <span class="dm-stat-label">Active</span>
                    </div>
                    <div class="dm-stat">
                        <span class="dm-stat-value" id="stat-excluded-claims">-</span>
                        <span class="dm-stat-label">Excluded</span>
                    </div>
                    <div class="dm-stat">
                        <span class="dm-stat-value" id="stat-sources">-</span>
                        <span class="dm-stat-label">Sources</span>
                    </div>
                    <div class="dm-stat">
                        <span class="dm-stat-value" id="stat-embeddings">-</span>
                        <span class="dm-stat-label">Embeddings</span>
                    </div>
                </div>
                
                <div class="dm-panels">
                    <!-- Sources Panel -->
                    <div class="dm-panel dm-sources-panel">
                        <div class="dm-panel-header">
                            <h3>📁 Source Files</h3>
                            <div class="dm-panel-actions">
                                <button class="dm-btn dm-btn-small" id="refresh-sources-btn" title="Refresh">🔄</button>
                            </div>
                        </div>
                        
                        <div class="dm-search-bar">
                            <input type="text" id="source-search-input" placeholder="Filter sources..." class="dm-input">
                        </div>
                        
                        <div class="dm-sources-list" id="sources-list">
                            <div class="dm-loading">Loading sources...</div>
                        </div>
                        
                        <div class="dm-panel-footer">
                            <span id="sources-selected-count">0 selected</span>
                            <button class="dm-btn dm-btn-danger dm-btn-small" id="delete-sources-btn" disabled>Delete Selected</button>
                        </div>
                    </div>
                    
                    <!-- Claims Panel -->
                    <div class="dm-panel dm-claims-panel">
                        <div class="dm-panel-header">
                            <h3>📄 Claims Browser</h3>
                            <div class="dm-panel-actions">
                                <button class="dm-btn dm-btn-small" id="refresh-claims-btn" title="Refresh">🔄</button>
                            </div>
                        </div>
                        
                        <div class="dm-filters-bar">
                            <select id="filter-source" class="dm-select">
                                <option value="">All Sources</option>
                            </select>
                            <select id="filter-type" class="dm-select">
                                <option value="">All Types</option>
                                <option value="PRIMARY">PRIMARY</option>
                                <option value="META">META</option>
                                <option value="SECONDARY">SECONDARY</option>
                                <option value="CONTEXTUAL">CONTEXTUAL</option>
                            </select>
                            <select id="filter-status" class="dm-select">
                                <option value="active">Active Only</option>
                                <option value="excluded">Excluded Only</option>
                                <option value="all">All</option>
                            </select>
                            <input type="text" id="claim-search-input" placeholder="Search claims..." class="dm-input dm-search-input">
                            <button class="dm-btn dm-btn-small" id="search-claims-btn">Search</button>
                        </div>
                        
                        <div class="dm-claims-list" id="claims-list">
                            <div class="dm-loading">Select a source or search to view claims</div>
                        </div>
                        
                        <div class="dm-pagination" id="claims-pagination">
                            <button class="dm-btn dm-btn-small" id="prev-page-btn" disabled>← Prev</button>
                            <span id="pagination-info">-</span>
                            <button class="dm-btn dm-btn-small" id="next-page-btn" disabled>Next →</button>
                        </div>
                        
                        <div class="dm-panel-footer">
                            <div class="dm-selection-info">
                                <input type="checkbox" id="select-all-claims" title="Select all on page">
                                <span id="claims-selected-count">0 selected</span>
                            </div>
                            <div class="dm-bulk-actions">
                                <button class="dm-btn dm-btn-warning dm-btn-small" id="exclude-claims-btn" disabled>Exclude</button>
                                <button class="dm-btn dm-btn-success dm-btn-small" id="restore-claims-btn" disabled>Restore</button>
                                <button class="dm-btn dm-btn-danger dm-btn-small" id="delete-claims-btn" disabled>Delete</button>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Claim Detail Panel -->
                    <div class="dm-panel dm-detail-panel">
                        <div class="dm-panel-header">
                            <h3>🔍 Claim Detail</h3>
                        </div>
                        
                        <div class="dm-detail-content" id="claim-detail">
                            <div class="dm-placeholder">Click a claim to view details</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Confirmation Modal -->
            <div class="dm-modal" id="dm-confirm-modal" style="display: none;">
                <div class="dm-modal-content">
                    <div class="dm-modal-header" id="modal-header">Confirm Action</div>
                    <div class="dm-modal-body" id="modal-body">Are you sure?</div>
                    <div class="dm-modal-footer">
                        <button class="dm-btn" id="modal-cancel-btn">Cancel</button>
                        <button class="dm-btn dm-btn-danger" id="modal-confirm-btn">Confirm</button>
                    </div>
                </div>
            </div>
        `;
    }
    
    bindEvents() {
        // Source panel events
        this.container.querySelector('#refresh-sources-btn').addEventListener('click', () => this.loadSources());
        this.container.querySelector('#source-search-input').addEventListener('input', (e) => this.filterSources(e.target.value));
        this.container.querySelector('#delete-sources-btn').addEventListener('click', () => this.confirmDeleteSources());
        
        // Claims panel events
        this.container.querySelector('#refresh-claims-btn').addEventListener('click', () => this.loadClaims());
        this.container.querySelector('#search-claims-btn').addEventListener('click', () => this.searchClaims());
        this.container.querySelector('#claim-search-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.searchClaims();
        });
        
        // Filter changes
        this.container.querySelector('#filter-source').addEventListener('change', (e) => {
            this.filters.source = e.target.value || null;
            this.claimsOffset = 0;
            this.loadClaims();
        });
        this.container.querySelector('#filter-type').addEventListener('change', (e) => {
            this.filters.claim_type = e.target.value || null;
            this.claimsOffset = 0;
            this.loadClaims();
        });
        this.container.querySelector('#filter-status').addEventListener('change', (e) => {
            this.filters.status = e.target.value;
            this.claimsOffset = 0;
            this.loadClaims();
        });
        
        // Pagination
        this.container.querySelector('#prev-page-btn').addEventListener('click', () => this.prevPage());
        this.container.querySelector('#next-page-btn').addEventListener('click', () => this.nextPage());
        
        // Bulk actions
        this.container.querySelector('#select-all-claims').addEventListener('change', (e) => this.toggleSelectAll(e.target.checked));
        this.container.querySelector('#exclude-claims-btn').addEventListener('click', () => this.confirmExcludeClaims());
        this.container.querySelector('#restore-claims-btn').addEventListener('click', () => this.restoreClaims());
        this.container.querySelector('#delete-claims-btn').addEventListener('click', () => this.confirmDeleteClaims());
        
        // Modal events
        this.container.querySelector('#modal-cancel-btn').addEventListener('click', () => this.hideModal());
        this.container.querySelector('#dm-confirm-modal').addEventListener('click', (e) => {
            if (e.target.id === 'dm-confirm-modal') this.hideModal();
        });
    }
    
    // =====================================================
    // Data Loading
    // =====================================================
    
    async loadStats() {
        try {
            const response = await fetch(`${this.apiBase}/api/admin/data/stats`);
            const data = await response.json();
            
            if (data.success) {
                const stats = data.stats;
                this.container.querySelector('#stat-total-claims').textContent = this.formatNumber(stats.total_claims);
                this.container.querySelector('#stat-active-claims').textContent = this.formatNumber(stats.active_claims);
                this.container.querySelector('#stat-excluded-claims').textContent = this.formatNumber(stats.excluded_claims);
                this.container.querySelector('#stat-sources').textContent = this.formatNumber(stats.total_sources);
                this.container.querySelector('#stat-embeddings').textContent = this.formatNumber(stats.total_embeddings || 0);
            }
        } catch (err) {
            console.error('Error loading stats:', err);
        }
    }
    
    async loadSources() {
        const listEl = this.container.querySelector('#sources-list');
        listEl.innerHTML = '<div class="dm-loading">Loading sources...</div>';
        
        try {
            const response = await fetch(`${this.apiBase}/api/admin/data/sources`);
            const data = await response.json();
            
            if (data.success) {
                this.sourcesData = data.sources;
                this.renderSources(data.sources);
                this.updateSourcesDropdown(data.sources);
            }
        } catch (err) {
            console.error('Error loading sources:', err);
            listEl.innerHTML = '<div class="dm-error">Error loading sources</div>';
        }
    }
    
    async loadClaims() {
        const listEl = this.container.querySelector('#claims-list');
        listEl.innerHTML = '<div class="dm-loading">Loading claims...</div>';
        
        try {
            const params = new URLSearchParams();
            if (this.filters.source) params.append('source', this.filters.source);
            if (this.filters.claim_type) params.append('claim_type', this.filters.claim_type);
            if (this.filters.status) params.append('status', this.filters.status);
            if (this.filters.search) params.append('search', this.filters.search);
            params.append('limit', this.claimsLimit);
            params.append('offset', this.claimsOffset);
            
            const response = await fetch(`${this.apiBase}/api/admin/data/claims?${params}`);
            const data = await response.json();
            
            if (data.success) {
                this.claimsData = data.claims;
                this.claimsTotal = data.total;
                this.renderClaims(data.claims);
                this.updatePagination();
            }
        } catch (err) {
            console.error('Error loading claims:', err);
            listEl.innerHTML = '<div class="dm-error">Error loading claims</div>';
        }
    }
    
    async loadClaimDetail(claimId) {
        const detailEl = this.container.querySelector('#claim-detail');
        detailEl.innerHTML = '<div class="dm-loading">Loading...</div>';
        
        try {
            const response = await fetch(`${this.apiBase}/api/admin/data/claims/${claimId}`);
            const data = await response.json();
            
            if (data.success) {
                this.currentClaim = data.claim;
                this.renderClaimDetail(data.claim);
            }
        } catch (err) {
            console.error('Error loading claim detail:', err);
            detailEl.innerHTML = '<div class="dm-error">Error loading claim details</div>';
        }
    }
    
    // =====================================================
    // Rendering
    // =====================================================
    
    renderSources(sources) {
        const listEl = this.container.querySelector('#sources-list');
        
        if (!sources || sources.length === 0) {
            listEl.innerHTML = '<div class="dm-placeholder">No sources found</div>';
            return;
        }
        
        listEl.innerHTML = sources.map(s => `
            <div class="dm-source-item ${this.selectedSources.has(s.source_file) ? 'selected' : ''}" 
                 data-source="${this.escapeHtml(s.source_file)}">
                <input type="checkbox" class="dm-source-checkbox" 
                       ${this.selectedSources.has(s.source_file) ? 'checked' : ''}>
                <div class="dm-source-info">
                    <div class="dm-source-name" title="${this.escapeHtml(s.source_file)}">
                        ${this.escapeHtml(s.display_name)}
                    </div>
                    <div class="dm-source-meta">
                        <span class="dm-badge">${s.claim_count} claims</span>
                        ${s.excluded_count > 0 ? `<span class="dm-badge dm-badge-warning">${s.excluded_count} excluded</span>` : ''}
                    </div>
                </div>
            </div>
        `).join('');
        
        // Bind click events
        listEl.querySelectorAll('.dm-source-item').forEach(item => {
            item.addEventListener('click', (e) => {
                if (e.target.type !== 'checkbox') {
                    this.selectSourceForFilter(item.dataset.source);
                }
            });
            
            item.querySelector('.dm-source-checkbox').addEventListener('change', (e) => {
                e.stopPropagation();
                this.toggleSourceSelection(item.dataset.source, e.target.checked);
            });
        });
    }
    
    renderClaims(claims) {
        const listEl = this.container.querySelector('#claims-list');
        
        if (!claims || claims.length === 0) {
            listEl.innerHTML = '<div class="dm-placeholder">No claims found</div>';
            return;
        }
        
        listEl.innerHTML = claims.map(c => `
            <div class="dm-claim-item ${c.excluded ? 'excluded' : ''} ${this.selectedClaims.has(c.claim_id) ? 'selected' : ''}"
                 data-claim-id="${c.claim_id}">
                <input type="checkbox" class="dm-claim-checkbox"
                       ${this.selectedClaims.has(c.claim_id) ? 'checked' : ''}>
                <div class="dm-claim-info">
                    <div class="dm-claim-text">${this.escapeHtml(this.truncate(c.claim_text, 120))}</div>
                    <div class="dm-claim-meta">
                        <span class="dm-badge dm-badge-type-${c.claim_type?.toLowerCase()}">${c.claim_type || 'UNKNOWN'}</span>
                        <span class="dm-badge ${c.excluded ? 'dm-badge-warning' : 'dm-badge-success'}">${c.status}</span>
                        <span class="dm-source-tag">${this.escapeHtml(c.display_source)}</span>
                    </div>
                </div>
            </div>
        `).join('');
        
        // Bind click events
        listEl.querySelectorAll('.dm-claim-item').forEach(item => {
            item.addEventListener('click', (e) => {
                if (e.target.type !== 'checkbox') {
                    this.loadClaimDetail(item.dataset.claimId);
                    listEl.querySelectorAll('.dm-claim-item').forEach(i => i.classList.remove('active'));
                    item.classList.add('active');
                }
            });
            
            item.querySelector('.dm-claim-checkbox').addEventListener('change', (e) => {
                e.stopPropagation();
                this.toggleClaimSelection(item.dataset.claimId, e.target.checked);
            });
        });
        
        // Reset select all checkbox
        this.container.querySelector('#select-all-claims').checked = false;
    }
    
    renderClaimDetail(claim) {
        const detailEl = this.container.querySelector('#claim-detail');
        
        const entitiesHtml = claim.entities?.length 
            ? claim.entities.map(e => `<span class="dm-badge">${this.escapeHtml(e.name)} (${e.type})</span>`).join(' ')
            : '<span class="dm-muted">None</span>';
        
        const temporalHtml = claim.temporal?.length
            ? claim.temporal.map(t => `<span class="dm-badge">${this.escapeHtml(t.date)}</span>`).join(' ')
            : '<span class="dm-muted">None</span>';
        
        const geoHtml = claim.geographic?.length
            ? claim.geographic.map(g => `<span class="dm-badge">${this.escapeHtml(g.location)}</span>`).join(' ')
            : '<span class="dm-muted">None</span>';
        
        detailEl.innerHTML = `
            <div class="dm-detail-section">
                <div class="dm-detail-label">Claim ID</div>
                <div class="dm-detail-value dm-mono">${claim.claim_id}</div>
            </div>
            
            <div class="dm-detail-section">
                <div class="dm-detail-label">Status</div>
                <div class="dm-detail-value">
                    <span class="dm-badge ${claim.excluded ? 'dm-badge-warning' : 'dm-badge-success'}">${claim.status}</span>
                    <span class="dm-badge dm-badge-type-${claim.claim_type?.toLowerCase()}">${claim.claim_type}</span>
                </div>
            </div>
            
            <div class="dm-detail-section">
                <div class="dm-detail-label">Source</div>
                <div class="dm-detail-value">${this.escapeHtml(claim.display_source)}</div>
            </div>
            
            <div class="dm-detail-section">
                <div class="dm-detail-label">Full Text</div>
                <div class="dm-detail-value dm-claim-full-text">${this.escapeHtml(claim.claim_text)}</div>
            </div>
            
            <div class="dm-detail-section">
                <div class="dm-detail-label">Entities</div>
                <div class="dm-detail-value">${entitiesHtml}</div>
            </div>
            
            <div class="dm-detail-section">
                <div class="dm-detail-label">Temporal</div>
                <div class="dm-detail-value">${temporalHtml}</div>
            </div>
            
            <div class="dm-detail-section">
                <div class="dm-detail-label">Geographic</div>
                <div class="dm-detail-value">${geoHtml}</div>
            </div>
            
            <div class="dm-detail-actions">
                ${claim.excluded 
                    ? `<button class="dm-btn dm-btn-success" onclick="window.dataManagement.restoreSingleClaim('${claim.claim_id}')">Restore</button>`
                    : `<button class="dm-btn dm-btn-warning" onclick="window.dataManagement.excludeSingleClaim('${claim.claim_id}')">Exclude</button>`
                }
                <button class="dm-btn dm-btn-danger" onclick="window.dataManagement.confirmDeleteSingleClaim('${claim.claim_id}')">Delete</button>
            </div>
        `;
    }
    
    updateSourcesDropdown(sources) {
        const dropdown = this.container.querySelector('#filter-source');
        const currentValue = dropdown.value;
        
        dropdown.innerHTML = '<option value="">All Sources</option>' +
            sources.map(s => `<option value="${this.escapeHtml(s.display_name)}">${this.escapeHtml(s.display_name)} (${s.claim_count})</option>`).join('');
        
        dropdown.value = currentValue;
    }
    
    updatePagination() {
        const start = this.claimsOffset + 1;
        const end = Math.min(this.claimsOffset + this.claimsData.length, this.claimsTotal);
        
        this.container.querySelector('#pagination-info').textContent = 
            `${start}-${end} of ${this.formatNumber(this.claimsTotal)}`;
        
        this.container.querySelector('#prev-page-btn').disabled = this.claimsOffset === 0;
        this.container.querySelector('#next-page-btn').disabled = end >= this.claimsTotal;
    }
    
    // =====================================================
    // Selection Handling
    // =====================================================
    
    toggleSourceSelection(sourceFile, selected) {
        if (selected) {
            this.selectedSources.add(sourceFile);
        } else {
            this.selectedSources.delete(sourceFile);
        }
        this.updateSourcesSelectionUI();
    }
    
    toggleClaimSelection(claimId, selected) {
        if (selected) {
            this.selectedClaims.add(claimId);
        } else {
            this.selectedClaims.delete(claimId);
        }
        this.updateClaimsSelectionUI();
    }
    
    toggleSelectAll(selected) {
        if (selected) {
            this.claimsData.forEach(c => this.selectedClaims.add(c.claim_id));
        } else {
            this.claimsData.forEach(c => this.selectedClaims.delete(c.claim_id));
        }
        
        // Update checkboxes
        this.container.querySelectorAll('.dm-claim-checkbox').forEach(cb => {
            cb.checked = selected;
        });
        this.container.querySelectorAll('.dm-claim-item').forEach(item => {
            item.classList.toggle('selected', selected);
        });
        
        this.updateClaimsSelectionUI();
    }
    
    updateSourcesSelectionUI() {
        this.container.querySelector('#sources-selected-count').textContent = `${this.selectedSources.size} selected`;
        this.container.querySelector('#delete-sources-btn').disabled = this.selectedSources.size === 0;
    }
    
    updateClaimsSelectionUI() {
        this.container.querySelector('#claims-selected-count').textContent = `${this.selectedClaims.size} selected`;
        
        const hasSelection = this.selectedClaims.size > 0;
        this.container.querySelector('#exclude-claims-btn').disabled = !hasSelection;
        this.container.querySelector('#restore-claims-btn').disabled = !hasSelection;
        this.container.querySelector('#delete-claims-btn').disabled = !hasSelection;
    }
    
    selectSourceForFilter(sourceFile) {
        // Find the source data to get display name
        const source = this.sourcesData.find(s => s.source_file === sourceFile);
        if (source) {
            this.filters.source = source.display_name;
            this.container.querySelector('#filter-source').value = source.display_name;
            this.claimsOffset = 0;
            this.loadClaims();
        }
    }
    
    // =====================================================
    // Actions
    // =====================================================
    
    searchClaims() {
        this.filters.search = this.container.querySelector('#claim-search-input').value;
        this.claimsOffset = 0;
        this.loadClaims();
    }
    
    filterSources(searchText) {
        const items = this.container.querySelectorAll('.dm-source-item');
        const search = searchText.toLowerCase();
        
        items.forEach(item => {
            const name = item.dataset.source.toLowerCase();
            item.style.display = name.includes(search) ? '' : 'none';
        });
    }
    
    prevPage() {
        if (this.claimsOffset > 0) {
            this.claimsOffset = Math.max(0, this.claimsOffset - this.claimsLimit);
            this.loadClaims();
        }
    }
    
    nextPage() {
        if (this.claimsOffset + this.claimsLimit < this.claimsTotal) {
            this.claimsOffset += this.claimsLimit;
            this.loadClaims();
        }
    }
    
    // =====================================================
    // API Actions
    // =====================================================
    
    async excludeSingleClaim(claimId) {
        await this.excludeClaims([claimId]);
    }
    
    async restoreSingleClaim(claimId) {
        await this.doRestoreClaims([claimId]);
    }
    
    confirmDeleteSingleClaim(claimId) {
        this.showModal(
            '🗑️ Delete Claim?',
            `<p>This will permanently delete this claim from:</p>
             <ul>
                <li>Neo4j knowledge graph</li>
                <li>PostgreSQL embeddings</li>
             </ul>
             <p><strong>This cannot be undone!</strong></p>`,
            () => this.doDeleteClaims([claimId])
        );
    }
    
    confirmExcludeClaims() {
        const count = this.selectedClaims.size;
        this.showModal(
            `⚠️ Exclude ${count} Claims?`,
            `<p>These claims will be hidden from detection analysis but can be restored later.</p>`,
            () => this.excludeClaims(Array.from(this.selectedClaims))
        );
    }
    
    async excludeClaims(claimIds) {
        try {
            const response = await fetch(`${this.apiBase}/api/admin/data/claims/exclude`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ claim_ids: claimIds })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showToast(`Excluded ${data.excluded_count} claims`, 'success');
                this.selectedClaims.clear();
                this.updateClaimsSelectionUI();
                this.loadClaims();
                this.loadStats();
                this.hideModal();
            } else {
                this.showToast('Error excluding claims', 'error');
            }
        } catch (err) {
            console.error('Error excluding claims:', err);
            this.showToast('Error excluding claims', 'error');
        }
    }
    
    async restoreClaims() {
        await this.doRestoreClaims(Array.from(this.selectedClaims));
    }
    
    async doRestoreClaims(claimIds) {
        try {
            const response = await fetch(`${this.apiBase}/api/admin/data/claims/restore`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ claim_ids: claimIds })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showToast(`Restored ${data.restored_count} claims`, 'success');
                this.selectedClaims.clear();
                this.updateClaimsSelectionUI();
                this.loadClaims();
                this.loadStats();
            } else {
                this.showToast('Error restoring claims', 'error');
            }
        } catch (err) {
            console.error('Error restoring claims:', err);
            this.showToast('Error restoring claims', 'error');
        }
    }
    
    confirmDeleteClaims() {
        const count = this.selectedClaims.size;
        this.showModal(
            `🗑️ Delete ${count} Claims?`,
            `<p>This will permanently delete these claims from:</p>
             <ul>
                <li>Neo4j knowledge graph</li>
                <li>PostgreSQL embeddings</li>
             </ul>
             <p><strong>This cannot be undone!</strong></p>`,
            () => this.doDeleteClaims(Array.from(this.selectedClaims))
        );
    }
    
    async doDeleteClaims(claimIds) {
        try {
            const response = await fetch(`${this.apiBase}/api/admin/data/claims`, {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ claim_ids: claimIds })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showToast(`Deleted ${data.deleted.neo4j_claims} claims`, 'success');
                this.selectedClaims.clear();
                this.updateClaimsSelectionUI();
                this.loadClaims();
                this.loadStats();
                this.loadSources();
                this.hideModal();
                
                // Clear detail panel if deleted claim was shown
                if (claimIds.includes(this.currentClaim?.claim_id)) {
                    this.container.querySelector('#claim-detail').innerHTML = 
                        '<div class="dm-placeholder">Claim deleted</div>';
                }
            } else {
                this.showToast('Error deleting claims', 'error');
            }
        } catch (err) {
            console.error('Error deleting claims:', err);
            this.showToast('Error deleting claims', 'error');
        }
    }
    
    confirmDeleteSources() {
        const count = this.selectedSources.size;
        const sources = Array.from(this.selectedSources);
        
        this.showModal(
            `🗑️ Delete ${count} Source(s)?`,
            `<p>This will permanently delete all claims from:</p>
             <ul>${sources.slice(0, 5).map(s => `<li>${this.escapeHtml(s.split('/').pop())}</li>`).join('')}</ul>
             ${sources.length > 5 ? `<p>...and ${sources.length - 5} more</p>` : ''}
             <p><strong>This cannot be undone!</strong></p>`,
            () => this.doDeleteSources(sources)
        );
    }
    
    async doDeleteSources(sources) {
        let totalDeleted = 0;
        
        for (const source of sources) {
            try {
                const response = await fetch(`${this.apiBase}/api/admin/data/sources`, {
                    method: 'DELETE',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ source_file: source })
                });
                
                const data = await response.json();
                if (data.success) {
                    totalDeleted += data.deleted.claims;
                }
            } catch (err) {
                console.error(`Error deleting source ${source}:`, err);
            }
        }
        
        this.showToast(`Deleted ${totalDeleted} claims from ${sources.length} sources`, 'success');
        this.selectedSources.clear();
        this.updateSourcesSelectionUI();
        this.loadSources();
        this.loadStats();
        this.loadClaims();
        this.hideModal();
    }
    
    // =====================================================
    // Modal Handling
    // =====================================================
    
    showModal(title, body, onConfirm) {
        const modal = this.container.querySelector('#dm-confirm-modal');
        modal.querySelector('#modal-header').textContent = title;
        modal.querySelector('#modal-body').innerHTML = body;
        
        const confirmBtn = modal.querySelector('#modal-confirm-btn');
        confirmBtn.onclick = () => {
            onConfirm();
        };
        
        modal.style.display = 'flex';
    }
    
    hideModal() {
        this.container.querySelector('#dm-confirm-modal').style.display = 'none';
    }
    
    showToast(message, type = 'info') {
        // Simple toast notification
        const toast = document.createElement('div');
        toast.className = `dm-toast dm-toast-${type}`;
        toast.textContent = message;
        document.body.appendChild(toast);
        
        setTimeout(() => toast.classList.add('show'), 10);
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
    
    // =====================================================
    // Utilities
    // =====================================================
    
    escapeHtml(str) {
        if (!str) return '';
        return str.replace(/[&<>"']/g, (m) => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[m]));
    }
    
    truncate(str, len) {
        if (!str) return '';
        return str.length > len ? str.slice(0, len) + '...' : str;
    }
    
    formatNumber(num) {
        if (num === null || num === undefined) return '-';
        return num.toLocaleString();
    }
}

// Export for global access (needed for onclick handlers in rendered HTML)
window.DataManagement = DataManagement;
