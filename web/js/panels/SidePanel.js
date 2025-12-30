/**
 * Aegis Insight v2.0 - Side Panel
 * Collapsible panel for displaying node details and connections
 */

import { CONFIG } from '../config.js';

export class SidePanel {
    constructor(containerId, graphRenderer) {
        this.container = document.querySelector(containerId);
        if (!this.container) {
            throw new Error(`Side panel container ${containerId} not found`);
        }
        
        this.graphRenderer = graphRenderer;
        this.currentNode = null;
        this.currentTab = 'details';
        
        this.initialize();
        this.attachEventListeners();
    }
    
    /**
     * Initialize panel HTML structure
     */
    initialize() {
        this.container.innerHTML = `
            <div class="panel-header">
                <h3 class="panel-title">Node Details</h3>
                <button class="panel-close" title="Close (ESC)">×</button>
            </div>
            
            <div class="panel-tabs">
                <button class="panel-tab active" data-tab="details">Details</button>
                <button class="panel-tab" data-tab="connections">Connections</button>
                <button class="panel-tab" data-tab="timeline">Timeline</button>
            </div>
            
            <div class="panel-content">
                <div id="tab-details" class="panel-tab-content active">
                    <!-- Details content will be rendered here -->
                </div>
                <div id="tab-connections" class="panel-tab-content">
                    <!-- Connections content will be rendered here -->
                </div>
                <div id="tab-timeline" class="panel-tab-content">
                    <!-- Timeline content will be rendered here -->
                </div>
            </div>
        `;
    }
    
    /**
     * Attach event listeners
     */
    attachEventListeners() {
        // Close button
        const closeBtn = this.container.querySelector('.panel-close');
        closeBtn.addEventListener('click', () => this.hide());
        
        // Tab switching
        const tabs = this.container.querySelectorAll('.panel-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const tabName = tab.dataset.tab;
                this.switchTab(tabName);
            });
        });
        
        // ESC key to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isVisible()) {
                this.hide();
            }
        });
        
        // Click outside to close (optional)
        document.addEventListener('click', (e) => {
            if (this.isVisible() && 
                !this.container.contains(e.target) && 
                !e.target.closest('.node')) {
                this.hide();
            }
        });
    }
    
    /**
     * Show panel with node details
     * @param {Object} node - Node data object
     */
    show(node) {
        if (!node) return;
        
        this.currentNode = node;
        this.container.classList.add('visible');
        
        // Update panel title with node type
        const title = this.container.querySelector('.panel-title');
        const nodeType = (node.type || 'Node').charAt(0).toUpperCase() + 
                        (node.type || 'Node').slice(1);
        title.textContent = `${nodeType} Details`;
        
        // Render content for current tab
        this.renderCurrentTab();
        
        // Highlight connections in graph
        if (this.graphRenderer) {
            this.graphRenderer.highlightConnections(node.id);
        }
    }
    
    /**
     * Hide panel
     */
    hide() {
        this.container.classList.remove('visible');
        this.currentNode = null;
        
        // Clear graph highlighting
        if (this.graphRenderer) {
            this.graphRenderer.clearHighlight();
            this.graphRenderer.clearSelection();
        }
    }
    
    /**
     * Check if panel is visible
     * @returns {boolean}
     */
    isVisible() {
        return this.container.classList.contains('visible');
    }
    
    /**
     * Switch to a different tab
     * @param {string} tabName - Tab name (details/connections/timeline)
     */
    switchTab(tabName) {
        this.currentTab = tabName;
        
        // Update tab buttons
        const tabs = this.container.querySelectorAll('.panel-tab');
        tabs.forEach(tab => {
            if (tab.dataset.tab === tabName) {
                tab.classList.add('active');
            } else {
                tab.classList.remove('active');
            }
        });
        
        // Update tab content
        const contents = this.container.querySelectorAll('.panel-tab-content');
        contents.forEach(content => {
            if (content.id === `tab-${tabName}`) {
                content.classList.add('active');
            } else {
                content.classList.remove('active');
            }
        });
        
        // Render new tab content
        this.renderCurrentTab();
    }
    
    /**
     * Render content for currently active tab
     */
    renderCurrentTab() {
        if (!this.currentNode) return;
        
        switch(this.currentTab) {
            case 'details':
                this.renderDetailsTab();
                break;
            case 'connections':
                this.renderConnectionsTab();
                break;
            case 'timeline':
                this.renderTimelineTab();
                break;
        }
    }
    
    /**
     * Render Details tab
     */
    renderDetailsTab() {
        const container = this.container.querySelector('#tab-details');
        const node = this.currentNode;
        
        let html = '<div class="panel-section">';
        
        // Node ID
        html += `
            <div class="detail-row">
                <span class="detail-label">ID:</span>
                <span class="detail-value monospace">${this.truncate(node.id, 40)}</span>
            </div>
        `;
        
        // Type
        html += `
            <div class="detail-row">
                <span class="detail-label">Type:</span>
                <span class="detail-value">${node.type || 'Unknown'}</span>
            </div>
        `;
        
        // Confidence / Trust Score (moved before claim text)
        if (node.confidence !== undefined && node.confidence !== null) {
            const confidencePercent = (node.confidence * 100).toFixed(1);
            const confidenceClass = node.confidence >= 0.7 ? 'high' : 
                                   node.confidence >= 0.5 ? 'medium' : 'low';
            html += `
                <div class="detail-row">
                    <span class="detail-label">Confidence:</span>
                    <span class="detail-value confidence ${confidenceClass}">${confidencePercent}%</span>
                </div>
            `;
        } else if (node.trust_score !== undefined && node.trust_score !== null) {
            const trustPercent = (node.trust_score * 100).toFixed(1);
            const trustClass = node.trust_score >= 0.7 ? 'high' : 
                              node.trust_score >= 0.5 ? 'medium' : 'low';
            html += `
                <div class="detail-row">
                    <span class="detail-label">Trust Score:</span>
                    <span class="detail-value confidence ${trustClass}">${trustPercent}%</span>
                </div>
            `;
        }
        
        // Full Claim Text (after trust score) - This is the main content
        // Check multiple possible field names for claim text
        // Also check inside metadata object
        let claimText = node.claim_text || node.text || node.content || node.description || node.claim;

        // If not found directly, check in metadata (including full_text)
        if (!claimText && node.metadata) {
            claimText = node.metadata.full_text || node.metadata.claim_text || node.metadata.text || node.metadata.content;
        }

        if (claimText) {
            html += `
                <div class="detail-row full-width">
                    <span class="detail-label">Claim Detail:</span>
                    <div class="detail-value claim-text">${claimText}</div>
                </div>
            `;
            
            // Add source file directly after claim text
            const sourceFileInline = node.source_file || node.source || 
                                     (node.metadata && (node.metadata.source_file || node.metadata.source));
            if (sourceFileInline) {
                const filenameShort = sourceFileInline.split('/').pop();
                html += `
                    <div class="detail-row full-width" style="margin-top: 4px;">
                        <span class="detail-value" style="font-size: 0.75rem; color: #888; font-style: italic;">
                            📄 ${filenameShort}
                        </span>
                    </div>
                `;
            }
        } else if (node.name || node.label) {
            html += `
                <div class="detail-row full-width">
                    <span class="detail-label">Name:</span>
                    <span class="detail-value">${node.name || node.label}</span>
                </div>
            `;
        } else {
            // Debug: Show what fields are available
            html += `
                <div class="detail-row full-width">
                    <span class="detail-label" style="color: #ff9800;">⚠️ Debug Info:</span>
                    <div class="detail-value" style="font-size: 0.75rem;">
                        No claim_text field found. Available fields: ${Object.keys(node).join(', ')}
                        ${node.metadata ? '<br>Metadata fields: ' + Object.keys(node.metadata).join(', ') : ''}
                    </div>
                </div>
            `;
        }

        // Claim Type (also check metadata)
        const claimType = node.claim_type || (node.metadata && node.metadata.claim_type);
        if (claimType) {
            html += `
                <div class="detail-row">
                    <span class="detail-label">Claim Type:</span>
                    <span class="detail-value badge">${claimType}</span>
                </div>
            `;
        }

        // Suppression Score
        if (node.suppression_score && node.suppression_score > 0.3) {
            const suppressionPercent = (node.suppression_score * 100).toFixed(1);
            html += `
                <div class="detail-row warning">
                    <span class="detail-label">⚠️ Suppression:</span>
                    <span class="detail-value">${suppressionPercent}%</span>
                </div>
            `;
        }

        html += '</div>'; // Close panel-section

        // Temporal Data Section (check metadata first)
        const temporalData = this.parseJSON(
            (node.metadata && node.metadata.temporal_data) || node.temporal_data
        );
        // Check for actual temporal data (arrays format from extraction)
        const hasAbsDates = temporalData?.absolute_dates?.length > 0;
        const hasRelDates = temporalData?.relative_dates?.length > 0;
        const hasMarkers = temporalData?.temporal_markers?.length > 0;
        const hasLegacy = temporalData?.year || temporalData?.date || temporalData?.period;
        
        if (temporalData && (hasAbsDates || hasRelDates || hasMarkers || hasLegacy)) {
            html += '<div class="panel-section">';
            html += '<h4 class="section-title">Temporal Information</h4>';
            
            // Handle absolute_dates array
            if (hasAbsDates) {
                temporalData.absolute_dates.forEach(d => {
                    html += `
                        <div class="detail-row">
                            <span class="detail-label">Date:</span>
                            <span class="detail-value">${d.date || 'Unknown'}${d.context ? ' - ' + d.context : ''}</span>
                        </div>
                    `;
                });
            }
            
            // Handle relative_dates array
            if (hasRelDates) {
                temporalData.relative_dates.forEach(d => {
                    html += `
                        <div class="detail-row">
                            <span class="detail-label">Period:</span>
                            <span class="detail-value">${d.period || d.reference || 'Unknown'}</span>
                        </div>
                    `;
                });
            }
            
            // Handle temporal_markers array
            if (hasMarkers) {
                temporalData.temporal_markers.forEach(m => {
                    html += `
                        <div class="detail-row">
                            <span class="detail-label">Time Ref:</span>
                            <span class="detail-value">${m.text || m.type || 'Unknown'}</span>
                        </div>
                    `;
                });
            }
            
            // Legacy format support
            if (temporalData.year) {
                html += `<div class="detail-row"><span class="detail-label">Year:</span><span class="detail-value">${temporalData.year}</span></div>`;
            }
            if (temporalData.date) {
                html += `<div class="detail-row"><span class="detail-label">Date:</span><span class="detail-value">${temporalData.date}</span></div>`;
            }
            if (temporalData.period) {
                html += `<div class="detail-row"><span class="detail-label">Period:</span><span class="detail-value">${temporalData.period}</span></div>`;
            }
            
            html += '</div>';
        }

        // Geographic Data Section (check metadata first)
        const geoData = this.parseJSON(
            (node.metadata && node.metadata.geographic_data) || node.geographic_data
        );
        // Check for actual geo data (locations array from extraction)
        const hasLocations = geoData?.locations?.length > 0;
        const hasCulturalContext = geoData?.cultural_context?.length > 0;
        const hasLegacyGeo = geoData?.location || geoData?.coordinates || geoData?.country;
        
        if (geoData && (hasLocations || hasCulturalContext || hasLegacyGeo)) {
            html += '<div class="panel-section">';
            html += '<h4 class="section-title">Geographic Information</h4>';
            
            // Handle locations array
            if (hasLocations) {
                geoData.locations.forEach(loc => {
                    const locName = loc.name || 'Unknown';
                    const locType = loc.type ? ` (${loc.type})` : '';
                    const locCountry = loc.country && loc.country !== 'not specified' ? `, ${loc.country}` : '';
                    html += `
                        <div class="detail-row">
                            <span class="detail-label">Location:</span>
                            <span class="detail-value">${locName}${locType}${locCountry}</span>
                        </div>
                    `;
                });
            }
            
            // Handle cultural_context array
            if (hasCulturalContext) {
                geoData.cultural_context.forEach(ctx => {
                    html += `
                        <div class="detail-row">
                            <span class="detail-label">Context:</span>
                            <span class="detail-value">${ctx}</span>
                        </div>
                    `;
                });
            }
            
            // Legacy format support
            if (geoData.location) {
                html += `
                    <div class="detail-row">
                        <span class="detail-label">Location:</span>
                        <span class="detail-value">${geoData.location}</span>
                    </div>
                `;
            }
            if (geoData.coordinates) {
                html += `
                    <div class="detail-row">
                        <span class="detail-label">Coordinates:</span>
                        <span class="detail-value monospace">${geoData.coordinates}</span>
                    </div>
                `;
            }
            if (geoData.country) {
                html += `
                    <div class="detail-row">
                        <span class="detail-label">Country:</span>
                        <span class="detail-value">${geoData.country}</span>
                    </div>
                `;
            }

            html += '</div>';
        }

        // Citation/Attribution Section (check metadata first)
        const citationData = this.parseJSON(
            (node.metadata && node.metadata.citation_data) || node.citation_data
        );
        if (citationData && (citationData.source || citationData.author || citationData.attribution_chain)) {
            html += '<div class="panel-section">';
            html += '<h4 class="section-title">Attribution</h4>';

            if (citationData.author) {
                html += `
                    <div class="detail-row">
                        <span class="detail-label">Author:</span>
                        <span class="detail-value">${citationData.author}</span>
                    </div>
                `;
            }
            if (citationData.source) {
                html += `
                    <div class="detail-row">
                        <span class="detail-label">Source:</span>
                        <span class="detail-value">${citationData.source}</span>
                    </div>
                `;
            }
            if (citationData.attribution_chain && Array.isArray(citationData.attribution_chain) && citationData.attribution_chain.length > 0) {
                html += `
                    <div class="detail-row full-width">
                        <span class="detail-label">Attribution Chain:</span>
                        <div class="detail-value attribution-chain">
                            ${citationData.attribution_chain.map((item, idx) => {
                                // Handle object format from extraction
                                if (typeof item === 'object') {
                                    const name = item.source_name || item.name || 'Unknown';
                                    const type = item.source_type || item.type || '';
                                    const author = item.author && item.author !== 'unknown' ? ` by ${item.author}` : '';
                                    return `<div class="chain-item">
                                        <span class="chain-number">${idx + 1}.</span> ${name}${type ? ' (' + type + ')' : ''}${author}
                                    </div>`;
                                }
                                // String format
                                return `<div class="chain-item">
                                    <span class="chain-number">${idx + 1}.</span> ${item}
                                </div>`;
                            }).join('')}
                        </div>
                    </div>
                `;
            }

            html += '</div>';
        }

        // Source File Section (check metadata first)
        const sourceFile = (node.metadata && node.metadata.source_file) || node.source_file;
        if (sourceFile) {
            html += '<div class="panel-section">';
            html += '<h4 class="section-title">Source</h4>';

            const filename = sourceFile.split('/').pop();
            const filepath = sourceFile;

            html += `
                <div class="detail-row full-width">
                    <span class="detail-label">File:</span>
                    <span class="detail-value monospace" title="${filepath}">${filename}</span>
                </div>
            `;

            const chunkIndex = (node.metadata && node.metadata.chunk_index) || node.chunk_index;
            if (chunkIndex !== undefined) {
                html += `
                    <div class="detail-row">
                        <span class="detail-label">Chunk:</span>
                        <span class="detail-value">${chunkIndex}</span>
                    </div>
                `;
            }

            html += '</div>';
        }

        // Action Buttons (only for claim nodes)
        if (node.type === 'claim') {
            const claimId = node.claim_id || (node.id && node.id.replace('claim-', '').split(':').pop());
            if (claimId) {
                html += '<div class="panel-section action-buttons">';
                html += '<h4 class="section-title">⚡ Actions</h4>';
                html += `
                    <div class="detail-row action-row">
                        <button class="dm-btn dm-btn-warning dm-btn-small" id="exclude-claim-btn" data-claim-id="${claimId}" title="Exclude from detection analysis">
                            🚫 Exclude
                        </button>
                        <button class="dm-btn dm-btn-danger dm-btn-small" id="delete-claim-btn" data-claim-id="${claimId}" title="Permanently delete claim">
                            🗑️ Delete
                        </button>
                    </div>
                `;
                html += '</div>';
            }
        }

        container.innerHTML = html;
        
        // Bind action button events
        this.bindActionButtons();
    }
    
    /**
     * Bind action button event handlers
     */
    bindActionButtons() {
        const excludeBtn = this.container.querySelector('#exclude-claim-btn');
        const deleteBtn = this.container.querySelector('#delete-claim-btn');
        
        if (excludeBtn) {
            excludeBtn.addEventListener('click', async () => {
                const claimId = excludeBtn.dataset.claimId;
                if (confirm('Exclude this claim from detection analysis?')) {
                    await this.excludeClaim(claimId);
                }
            });
        }
        
        if (deleteBtn) {
            deleteBtn.addEventListener('click', async () => {
                const claimId = deleteBtn.dataset.claimId;
                if (confirm('⚠️ PERMANENTLY delete this claim? This cannot be undone!')) {
                    await this.deleteClaim(claimId);
                }
            });
        }
    }
    
    /**
     * Exclude claim from detection
     */
    async excludeClaim(claimId) {
        try {
            const response = await fetch('/api/admin/data/claims/exclude', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ claim_ids: [`claim_${claimId}`] })
            });
            
            const data = await response.json();
            if (data.success) {
                this.showToast('Claim excluded from detection', 'success');
                const btn = this.container.querySelector('#exclude-claim-btn');
                if (btn) {
                    btn.textContent = '✓ Excluded';
                    btn.disabled = true;
                }
            } else {
                this.showToast('Error: ' + (data.error || 'Unknown'), 'error');
            }
        } catch (err) {
            console.error('Error excluding claim:', err);
            this.showToast('Error excluding claim', 'error');
        }
    }
    
    /**
     * Delete claim permanently
     */
    async deleteClaim(claimId) {
        try {
            const response = await fetch('/api/admin/data/claims', {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ claim_ids: [`claim_${claimId}`] })
            });
            
            const data = await response.json();
            if (data.success) {
                this.showToast('Claim deleted', 'success');
                if (this.graphRenderer && this.currentNode) {
                    this.graphRenderer.removeNode(this.currentNode.id);
                }
                this.hide();
            } else {
                this.showToast('Error: ' + (data.error || 'Unknown'), 'error');
            }
        } catch (err) {
            console.error('Error deleting claim:', err);
            this.showToast('Error deleting claim', 'error');
        }
    }
    
    /**
     * Show toast notification
     */
    showToast(message, type = 'info') {
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

    /**
     * Render Connections tab
     */
    renderConnectionsTab() {
        const container = this.container.querySelector('#tab-connections');
        const node = this.currentNode;

        // Get connected nodes from graph data
        const connections = this.getConnectedNodes(node.id);

        let html = '<div class="panel-section">';

        if (connections.length === 0) {
            html += '<div class="empty-message">No connections found</div>';
        } else {
            html += `
                <div class="connection-stats">
                    <span class="stat-badge">${connections.length} Connected Node${connections.length !== 1 ? 's' : ''}</span>
                </div>
            `;

            html += '<div class="connection-list">';

            connections.forEach(conn => {
                const relationshipType = conn.relationshipType || 'RELATED_TO';
                const directionIcon = conn.direction === 'outgoing' ? '→' : '←';

                html += `
                    <div class="connection-item" data-node-id="${conn.node.id}">
                        <div class="connection-header">
                            <span class="connection-type">${directionIcon} ${relationshipType}</span>
                        </div>
                        <div class="connection-name">${conn.node.name || conn.node.label || conn.node.id}</div>
                        ${conn.node.claim_text ?
                            `<div class="connection-claim">${this.truncate(conn.node.claim_text, 100)}</div>` :
                            ''}
                    </div>
                `;
            });

            html += '</div>';
        }

        html += '</div>';

        container.innerHTML = html;

        // Add click handlers for connection items
        container.querySelectorAll('.connection-item').forEach(item => {
            item.addEventListener('click', (event) => {
                event.stopPropagation(); // Prevent click-outside handler from firing
                const nodeId = item.dataset.nodeId;
                const connectedNode = this.graphRenderer.data.nodes.find(n => n.id === nodeId);
                if (connectedNode) {
                    console.log('Navigating to connected node:', connectedNode.id);
                    // Switch back to Details tab and show the new node
                    this.switchTab('details');
                    this.show(connectedNode);
                }
            });
        });
    }

    /**
     * Render Timeline tab
     */
    renderTimelineTab() {
        const container = this.container.querySelector('#tab-timeline');
        const node = this.currentNode;

        let html = '<div class="panel-section">';

        const temporalData = this.parseJSON(node.temporal_data || (node.metadata && node.metadata.temporal_data));
        
        // Check for actual temporal content in various formats
        const hasAbsoluteDates = temporalData?.absolute_dates?.length > 0;
        const hasRelativeDates = temporalData?.relative_dates?.length > 0;
        const hasTemporalMarkers = temporalData?.temporal_markers?.length > 0;
        const hasSimpleDate = temporalData?.year || temporalData?.date || temporalData?.period;
        
        if (!temporalData || (!hasAbsoluteDates && !hasRelativeDates && !hasTemporalMarkers && !hasSimpleDate)) {
            html += '<div class="empty-message">No temporal data available</div>';
        } else {
            html += '<h4 class="section-title">📅 Timeline</h4>';
            html += '<div class="timeline">';
            
            // Handle absolute_dates array (extraction format)
            if (hasAbsoluteDates) {
                temporalData.absolute_dates.forEach(dateObj => {
                    const displayDate = dateObj.date || dateObj.year || 'Unknown';
                    const context = dateObj.context || '';
                    html += `
                        <div class="timeline-item">
                            <div class="timeline-marker"></div>
                            <div class="timeline-content">
                                <div class="timeline-date">${displayDate}</div>
                                ${context ? `<div class="timeline-description">${context}</div>` : ''}
                            </div>
                        </div>
                    `;
                });
            }
            
            // Handle relative_dates array
            if (hasRelativeDates) {
                temporalData.relative_dates.forEach(relDate => {
                    const period = relDate.period || relDate.reference || 'Period';
                    html += `
                        <div class="timeline-item">
                            <div class="timeline-marker period"></div>
                            <div class="timeline-content">
                                <div class="timeline-date">${period}</div>
                            </div>
                        </div>
                    `;
                });
            }
            
            // Handle temporal_markers array  
            if (hasTemporalMarkers) {
                temporalData.temporal_markers.forEach(marker => {
                    const text = marker.text || marker.type || 'Temporal ref';
                    html += `
                        <div class="timeline-item">
                            <div class="timeline-marker marker"></div>
                            <div class="timeline-content">
                                <div class="timeline-date">${text}</div>
                            </div>
                        </div>
                    `;
                });
            }
            
            // Legacy simple format
            if (temporalData.year || temporalData.date) {
                const displayDate = temporalData.date || temporalData.year;
                html += `
                    <div class="timeline-item">
                        <div class="timeline-marker"></div>
                        <div class="timeline-content">
                            <div class="timeline-date">${displayDate}</div>
                            <div class="timeline-description">${node.claim_text || node.name || 'Event occurred'}</div>
                        </div>
                    </div>
                `;
            }

            if (temporalData.period) {
                html += `
                    <div class="timeline-item">
                        <div class="timeline-marker period"></div>
                        <div class="timeline-content">
                            <div class="timeline-date">Period: ${temporalData.period}</div>
                        </div>
                    </div>
                `;
            }

            // Show related temporal nodes if any
            const connectedNodes = this.getConnectedNodes(node.id);
            const temporalConnections = connectedNodes.filter(conn => {
                const connTemporal = this.parseJSON(conn.node.temporal_data);
                return connTemporal && Object.keys(connTemporal).length > 0;
            }).sort((a, b) => {
                const aData = this.parseJSON(a.node.temporal_data);
                const bData = this.parseJSON(b.node.temporal_data);
                const aYear = aData.year || 0;
                const bYear = bData.year || 0;
                return aYear - bYear;
            });

            if (temporalConnections.length > 0) {
                html += '<div class="timeline-divider">Related Events</div>';

                temporalConnections.forEach(conn => {
                    const connTemporal = this.parseJSON(conn.node.temporal_data);
                    const displayDate = connTemporal.date || connTemporal.year || 'Unknown';

                    html += `
                        <div class="timeline-item related">
                            <div class="timeline-marker small"></div>
                            <div class="timeline-content">
                                <div class="timeline-date">${displayDate}</div>
                                <div class="timeline-description">${this.truncate(conn.node.claim_text || conn.node.name || '', 80)}</div>
                            </div>
                        </div>
                    `;
                });
            }

            html += '</div>'; // Close timeline
        }

        html += '</div>';

        container.innerHTML = html;
    }

    /**
     * Get nodes connected to a given node ID
     * @param {string} nodeId - Node ID
     * @returns {Array} Array of connection objects with node and relationship data
     */
    getConnectedNodes(nodeId) {
        if (!this.graphRenderer || !this.graphRenderer.data) return [];

        const connections = [];
        const links = this.graphRenderer.data.links || this.graphRenderer.data.edges || [];
        const nodes = this.graphRenderer.data.nodes || [];

        links.forEach(link => {
            // Outgoing connections
            const sourceId = typeof link.source === "object" ? link.source.id : link.source;
            const targetId = typeof link.target === "object" ? link.target.id : link.target;
            if (sourceId === nodeId) {
                connections.push({
                    node: link.target,
                    relationshipType: link.type || link.relationship_type || 'RELATED_TO',
                    direction: 'outgoing'
                });
            }
            // Incoming connections
            if (targetId === nodeId) {
                connections.push({
                    node: link.source,
                    relationshipType: link.type || link.relationship_type || 'RELATED_TO',
                    direction: 'incoming'
                });
            }
        });

        return connections;
    }

    /**
     * Parse JSON string safely
     * @param {string} jsonString - JSON string to parse
     * @returns {Object|null} Parsed object or null
     */
    parseJSON(jsonString) {
        if (!jsonString) return null;
        if (typeof jsonString === 'object') return jsonString;

        try {
            return JSON.parse(jsonString);
        } catch (e) {
            return null;
        }
    }

    /**
     * Truncate text to max length
     * @param {string} text - Text to truncate
     * @param {number} maxLength - Maximum length
     * @returns {string} Truncated text
     */
    truncate(text, maxLength = 50) {
        if (!text) return '';
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength - 3) + '...';
    }
}