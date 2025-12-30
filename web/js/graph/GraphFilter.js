/**
 * Aegis Insight v2.0 - Graph Filter & Clustering
 * 
 * Features:
 * - Filter Panel: node count, trust threshold, type toggles
 * - Node Clustering: group detected nodes, visual swarm effect
 * - Real-time filtering without re-fetching data
 * 
 * @author Aegis Development Team
 * @created 2025-12-07
 */

import { CONFIG, getScoreColor } from '../config.js';

// ============================================================================
// GRAPH FILTER - Main filtering logic
// ============================================================================

export class GraphFilter {
    constructor(graphRenderer, options = {}) {
        this.graph = graphRenderer;
        this.container = options.container || document.getElementById('graph-container');
        this.onFilterChange = options.onFilterChange || (() => {});
        
        // Original unfiltered data
        this.originalNodes = [];
        this.originalEdges = [];
        
        // Current filter state
        this.filters = {
            maxNodes: 100,
            trustThreshold: 0,
            showEntities: true,
            showClaims: true,
            showDocuments: true,
            showDetectedOnly: false,
            claimTypes: {
                PRIMARY: true,
                SECONDARY: true,
                META: true,
                CONTEXTUAL: true
            }
        };
        
        // Clustering state
        this.clusteringEnabled = false;
        this.clusterForce = null;
        
        // Narrative clustering state (U7)
        this.narrativeClusteringEnabled = false;
        this.narrativeStances = new Map();  // nodeId -> 'pro' | 'anti' | 'neutral'
        this.centralEntity = null;
        
        // Detection data
        this.detectedNodeIds = new Set();
        this.detectionScores = new Map();
        
        this.panel = null;
        this.isCollapsed = true;
    }
    
    /**
     * Initialize with graph data
     */
    setData(nodes, edges) {
        this.originalNodes = [...nodes];
        this.originalEdges = [...edges];
        this.applyFilters();
    }
    
    /**
     * Set detected nodes from detection results
     */
    setDetectionResults(results) {
        this.detectedNodeIds.clear();
        this.detectionScores.clear();
        
        // Handle various result formats
        const affectedClaims = results.affected_claims || results.analyzed_claims || results.coordinated_claims || results.anomalous_claims || [];
        
        affectedClaims.forEach(claim => {
            const id = claim.graph_id || claim.claim_id || claim.id;
            if (id) {
                this.detectedNodeIds.add(id);
                this.detectionScores.set(id, claim.indicator_score || claim.score || 0);
            }
        });
        
        // Update filter panel stats
        this.updateDetectionStats();
        
        // Re-apply filters if "show detected only" is enabled
        if (this.filters.showDetectedOnly) {
            this.applyFilters();
        }
    }
    
    /**
     * Create and inject the filter panel UI
     */
    createFilterPanel() {
        if (this.panel) return;
        
        this.panel = document.createElement('div');
        this.panel.className = 'graph-filter-panel collapsed';
        this.panel.innerHTML = `
            <div class="gf-header" id="gf-toggle">
                <span class="gf-icon">⚙️</span>
                <span class="gf-title">Filters</span>
                <span class="gf-collapse-icon">▶</span>
            </div>
            
            <div class="gf-body">
                <!-- Node Count Slider -->
                <div class="gf-group">
                    <label class="gf-label">
                        <span>Max Nodes</span>
                        <span class="gf-value" id="gf-node-count-value">${this.filters.maxNodes}</span>
                    </label>
                    <input type="range" id="gf-node-count" class="gf-slider"
                           min="10" max="500" step="10" value="${this.filters.maxNodes}">
                </div>
                
                <!-- Trust Threshold Slider -->
                <div class="gf-group">
                    <label class="gf-label">
                        <span>Min Trust</span>
                        <span class="gf-value" id="gf-trust-value">${this.filters.trustThreshold.toFixed(1)}</span>
                    </label>
                    <input type="range" id="gf-trust-threshold" class="gf-slider"
                           min="0" max="1" step="0.1" value="${this.filters.trustThreshold}">
                </div>
                
                <!-- Node Type Toggles -->
                <div class="gf-group">
                    <label class="gf-label">Node Types</label>
                    <div class="gf-toggles">
                        <label class="gf-toggle">
                            <input type="checkbox" id="gf-show-claims" checked>
                            <span class="gf-toggle-dot" style="background: #4CAF50;"></span>
                            Claims
                        </label>
                        <label class="gf-toggle">
                            <input type="checkbox" id="gf-show-entities" checked>
                            <span class="gf-toggle-dot" style="background: #00ccff;"></span>
                            Entities
                        </label>
                        <label class="gf-toggle">
                            <input type="checkbox" id="gf-show-documents" checked>
                            <span class="gf-toggle-dot" style="background: #607D8B;"></span>
                            Documents
                        </label>
                    </div>
                </div>
                
                <!-- Claim Type Toggles -->
                <div class="gf-group">
                    <label class="gf-label">Claim Types</label>
                    <div class="gf-toggles gf-claim-types">
                        <label class="gf-toggle">
                            <input type="checkbox" id="gf-claim-primary" checked>
                            <span class="gf-toggle-dot" style="background: #4CAF50;"></span>
                            PRIMARY
                        </label>
                        <label class="gf-toggle">
                            <input type="checkbox" id="gf-claim-secondary" checked>
                            <span class="gf-toggle-dot" style="background: #2196F3;"></span>
                            SECONDARY
                        </label>
                        <label class="gf-toggle">
                            <input type="checkbox" id="gf-claim-meta" checked>
                            <span class="gf-toggle-dot" style="background: #FF9800;"></span>
                            META
                        </label>
                        <label class="gf-toggle">
                            <input type="checkbox" id="gf-claim-contextual" checked>
                            <span class="gf-toggle-dot" style="background: #607D8B;"></span>
                            CONTEXTUAL
                        </label>
                    </div>
                </div>
                
                <!-- Detection Filter -->
                <div class="gf-group gf-detection-group">
                    <label class="gf-label">
                        <span>Detection</span>
                        <span class="gf-badge" id="gf-detected-count">0</span>
                    </label>
                    <label class="gf-toggle gf-toggle-highlight">
                        <input type="checkbox" id="gf-show-detected-only">
                        <span>Show Detected Only</span>
                    </label>
                </div>
                
                <!-- Clustering Control -->
                <div class="gf-group">
                    <label class="gf-label">Layout</label>
                    <label class="gf-toggle gf-toggle-highlight">
                        <input type="checkbox" id="gf-enable-clustering">
                        <span>Cluster Detected Nodes</span>
                    </label>
                    <label class="gf-toggle gf-toggle-highlight" style="margin-top: 6px;">
                        <input type="checkbox" id="gf-enable-narrative-clustering">
                        <span>Show Opposing Narratives</span>
                    </label>
                </div>
                
                <!-- Stats -->
                <div class="gf-stats">
                    <span id="gf-stats-text">Showing 0 of 0 nodes</span>
                </div>
                
                <!-- Reset Button -->
                <button class="gf-reset-btn" id="gf-reset">Reset Filters</button>
            </div>
        `;
        
        // Insert into container
        this.container.appendChild(this.panel);
        
        // Bind events
        this.bindPanelEvents();
    }
    
    /**
     * Bind filter panel events
     */
    bindPanelEvents() {
        // Toggle panel
        this.panel.querySelector('#gf-toggle').addEventListener('click', () => {
            this.isCollapsed = !this.isCollapsed;
            this.panel.classList.toggle('collapsed', this.isCollapsed);
            this.panel.querySelector('.gf-collapse-icon').textContent = this.isCollapsed ? '▶' : '▼';
        });
        
        // Node count slider
        const nodeCountSlider = this.panel.querySelector('#gf-node-count');
        nodeCountSlider.addEventListener('input', (e) => {
            this.filters.maxNodes = parseInt(e.target.value);
            this.panel.querySelector('#gf-node-count-value').textContent = this.filters.maxNodes;
            this.applyFilters();
        });
        
        // Trust threshold slider
        const trustSlider = this.panel.querySelector('#gf-trust-threshold');
        trustSlider.addEventListener('input', (e) => {
            this.filters.trustThreshold = parseFloat(e.target.value);
            this.panel.querySelector('#gf-trust-value').textContent = this.filters.trustThreshold.toFixed(1);
            this.applyFilters();
        });
        
        // Node type toggles
        this.panel.querySelector('#gf-show-claims').addEventListener('change', (e) => {
            this.filters.showClaims = e.target.checked;
            this.applyFilters();
        });
        this.panel.querySelector('#gf-show-entities').addEventListener('change', (e) => {
            this.filters.showEntities = e.target.checked;
            this.applyFilters();
        });
        this.panel.querySelector('#gf-show-documents').addEventListener('change', (e) => {
            this.filters.showDocuments = e.target.checked;
            this.applyFilters();
        });
        
        // Claim type toggles
        ['primary', 'secondary', 'meta', 'contextual'].forEach(type => {
            this.panel.querySelector(`#gf-claim-${type}`).addEventListener('change', (e) => {
                this.filters.claimTypes[type.toUpperCase()] = e.target.checked;
                this.applyFilters();
            });
        });
        
        // Detection filter
        this.panel.querySelector('#gf-show-detected-only').addEventListener('change', (e) => {
            this.filters.showDetectedOnly = e.target.checked;
            this.applyFilters();
        });
        
        // Clustering toggle
        this.panel.querySelector('#gf-enable-clustering').addEventListener('change', (e) => {
            this.clusteringEnabled = e.target.checked;
            // Disable narrative clustering if enabling detection clustering
            if (this.clusteringEnabled && this.narrativeClusteringEnabled) {
                this.narrativeClusteringEnabled = false;
                this.panel.querySelector('#gf-enable-narrative-clustering').checked = false;
                this.removeNarrativeClusteringForce();
            }
            this.applyFilters();
            if (this.clusteringEnabled) {
                this.applyClusteringForce();
            } else {
                this.removeClusteringForce();
            }
        });
        
        // Narrative clustering toggle (U7)
        this.panel.querySelector('#gf-enable-narrative-clustering').addEventListener('change', (e) => {
            this.narrativeClusteringEnabled = e.target.checked;
            // Disable detection clustering if enabling narrative clustering
            if (this.narrativeClusteringEnabled && this.clusteringEnabled) {
                this.clusteringEnabled = false;
                this.panel.querySelector('#gf-enable-clustering').checked = false;
                this.removeClusteringForce();
            }
            if (this.narrativeClusteringEnabled) {
                this.applyNarrativeClusteringForce();
            } else {
                this.removeNarrativeClusteringForce();
            }
        });
        
        // Reset button
        this.panel.querySelector('#gf-reset').addEventListener('click', () => {
            this.resetFilters();
        });
    }
    
    /**
     * Apply current filters to the graph
     */
    applyFilters() {
        let filteredNodes = [...this.originalNodes];
        
        // Filter 1: Node type
        filteredNodes = filteredNodes.filter(node => {
            const type = (node.type || 'claim').toLowerCase();
            if (type === 'entity' && !this.filters.showEntities) return false;
            if (type === 'claim' && !this.filters.showClaims) return false;
            if ((type === 'document' || type === 'source') && !this.filters.showDocuments) return false;
            return true;
        });
        
        // Filter 2: Claim type
        filteredNodes = filteredNodes.filter(node => {
            const type = (node.type || 'claim').toLowerCase();
            if (type !== 'claim') return true; // Non-claims pass through
            const claimType = (node.claim_type || 'UNKNOWN').toUpperCase();
            return this.filters.claimTypes[claimType] !== false;
        });
        
        // Filter 3: Trust threshold
        filteredNodes = filteredNodes.filter(node => {
            const trust = node.trust_score ?? 0.5;
            return trust >= this.filters.trustThreshold;
        });
        
        // Filter 4: Detection only
        if (this.filters.showDetectedOnly && this.detectedNodeIds.size > 0) {
            filteredNodes = filteredNodes.filter(node => 
                this.isNodeDetected(node)
            );
        }
        
        // Filter 5: Max nodes (prioritize detected nodes)
        if (filteredNodes.length > this.filters.maxNodes) {
            // Sort: detected nodes first, then by trust score
            filteredNodes.sort((a, b) => {
                const aDetected = this.isNodeDetected(a) ? 1 : 0;
                const bDetected = this.isNodeDetected(b) ? 1 : 0;
                if (aDetected !== bDetected) return bDetected - aDetected;
                return (b.trust_score || 0) - (a.trust_score || 0);
            });
            filteredNodes = filteredNodes.slice(0, this.filters.maxNodes);
        }
        
        // Filter edges to only include visible nodes
        const visibleNodeIds = new Set(filteredNodes.map(n => n.id));
        let filteredEdges = this.originalEdges.filter(edge => {
            const sourceId = edge.source?.id || edge.source;
            const targetId = edge.target?.id || edge.target;
            return visibleNodeIds.has(sourceId) && visibleNodeIds.has(targetId);
        });
        
        // Mark detected nodes
        filteredNodes.forEach(node => {
            node.isDetected = this.isNodeDetected(node);
            node.detectionScore = this.getDetectionScore(node);
        });
        
        // Update stats display
        this.updateStats(filteredNodes);
        
        // Notify callback
        this.onFilterChange(filteredNodes, filteredEdges);
        
        // Re-apply detection highlights after a brief delay (after render completes)
        if (this.detectedNodeIds.size > 0) {
            setTimeout(() => this.reapplyHighlights(), 100);
        }
        
        return { nodes: filteredNodes, edges: filteredEdges };
    }
    
    /**
     * Re-apply detection highlights after filter/render
     */
    reapplyHighlights() {
        // Access the detection highlighter via window.aegis
        const highlighter = window.aegis?.detectionHighlighter || 
                           window.aegis?.graphRenderer?.detectionHighlighter;
        
        if (highlighter && window.detectionControls?.lastResults) {
            const rawResults = window.detectionControls.lastResults;
            const results = rawResults.result || rawResults;
            const mode = window.detectionControls.activeMode || 'suppression';
            
            console.log('Re-applying highlights after filter');
            highlighter.applyHighlights(results, mode);
        }
    }
    
    /**
     * Check if a node matches any detected ID
     */
    isNodeDetected(node) {
        if (this.detectedNodeIds.has(node.id)) return true;
        
        // Try matching with various ID formats
        for (const detectedId of this.detectedNodeIds) {
            if (this.nodeMatchesId(node.id, detectedId)) return true;
        }
        return false;
    }
    
    /**
     * Flexible ID matching
     */
    nodeMatchesId(nodeId, targetId) {
        if (!nodeId || !targetId) return false;
        if (nodeId === targetId) return true;
        
        // Normalize IDs
        const normalizeId = (id) => {
            let n = String(id);
            if (n.startsWith('claim-')) n = n.substring(6);
            return n;
        };
        
        const normNode = normalizeId(nodeId);
        const normTarget = normalizeId(targetId);
        
        if (normNode === normTarget) return true;
        
        // Compare last segment
        const nodeNumeric = normNode.split(':').pop();
        const targetNumeric = normTarget.split(':').pop();
        
        return nodeNumeric === targetNumeric;
    }
    
    /**
     * Get detection score for a node
     */
    getDetectionScore(node) {
        if (this.detectionScores.has(node.id)) {
            return this.detectionScores.get(node.id);
        }
        
        for (const [detectedId, score] of this.detectionScores) {
            if (this.nodeMatchesId(node.id, detectedId)) {
                return score;
            }
        }
        return 0;
    }
    
    /**
     * Update stats display
     */
    updateStats(filteredNodes) {
        if (!this.panel) return;
        
        const statsEl = this.panel.querySelector('#gf-stats-text');
        if (statsEl) {
            const detected = filteredNodes.filter(n => n.isDetected).length;
            statsEl.textContent = `Showing ${filteredNodes.length} of ${this.originalNodes.length} nodes (${detected} detected)`;
        }
    }
    
    /**
     * Update detection stats badge
     */
    updateDetectionStats() {
        if (!this.panel) return;
        
        const badge = this.panel.querySelector('#gf-detected-count');
        if (badge) {
            badge.textContent = this.detectedNodeIds.size;
            badge.style.display = this.detectedNodeIds.size > 0 ? 'inline-block' : 'none';
        }
    }
    
    /**
     * Reset all filters to defaults
     */
    resetFilters() {
        // Broadcast reset event so other components (like timeline) can clean up
        document.dispatchEvent(new CustomEvent('aegis:filtersReset'));
        
        this.filters = {
            maxNodes: 100,
            trustThreshold: 0,
            showEntities: true,
            showClaims: true,
            showDocuments: true,
            showDetectedOnly: false,
            claimTypes: {
                PRIMARY: true,
                SECONDARY: true,
                META: true,
                CONTEXTUAL: true
            }
        };
        
        // Reset UI
        if (this.panel) {
            this.panel.querySelector('#gf-node-count').value = 100;
            this.panel.querySelector('#gf-node-count-value').textContent = '100';
            this.panel.querySelector('#gf-trust-threshold').value = 0;
            this.panel.querySelector('#gf-trust-value').textContent = '0.0';
            this.panel.querySelector('#gf-show-claims').checked = true;
            this.panel.querySelector('#gf-show-entities').checked = true;
            this.panel.querySelector('#gf-show-documents').checked = true;
            this.panel.querySelector('#gf-show-detected-only').checked = false;
            this.panel.querySelector('#gf-enable-clustering').checked = false;
            
            ['primary', 'secondary', 'meta', 'contextual'].forEach(type => {
                this.panel.querySelector(`#gf-claim-${type}`).checked = true;
            });
        }
        
        this.clusteringEnabled = false;
        this.narrativeClusteringEnabled = false;
        this.removeClusteringForce();
        if (this.removeNarrativeClusteringForce) {
            this.removeNarrativeClusteringForce();
        }
        
        // Reset narrative clustering UI toggle
        if (this.panel?.querySelector('#gf-enable-narrative-clustering')) {
            this.panel.querySelector('#gf-enable-narrative-clustering').checked = false;
        }
        
        // Deactivate timeline if it exists
        if (this.graph.timeline?.forceDeactivate) {
            this.graph.timeline.forceDeactivate();
        } else if (this.graph.timelineLayout?.forceDeactivate) {
            this.graph.timelineLayout.forceDeactivate();
        }
        // Also try to clean up timeline elements directly
        if (this.graph.g) {
            this.graph.g.selectAll('.timeline-layout').remove();
            this.graph.g.selectAll('.timeline-grid').remove();
        }
        d3.select('#timeline-controls').remove();
        
        this.applyFilters();
    }
    
    // ========================================================================
    // CLUSTERING / SWARM
    // ========================================================================
    
    /**
     * Apply clustering force to group detected nodes
     */
    applyClusteringForce() {
        const width = this.graph.width || 800;
        const height = this.graph.height || 600;
        
        // Get current nodes
        const nodes = this.graph.data?.nodes || [];
        if (nodes.length === 0) {
            console.warn('GraphFilter: No nodes to cluster');
            return;
        }
        
        // Only cluster detected nodes - leave others in natural layout
        const detectedNodes = nodes.filter(n => n.isDetected);
        const otherNodes = nodes.filter(n => !n.isDetected);
        const detectedIds = new Set(detectedNodes.map(n => n.id));
        
        console.log("Clustering:", detectedNodes.length, "detected nodes to left side");
        
        if (detectedNodes.length === 0) {
            console.warn('GraphFilter: No detected nodes to cluster');
            return;
        }
        
        // Stop simulation so it doesn't fight us
        if (this.graph.simulation) {
            this.graph.simulation.stop();
        }
        
        // Measure the tangle bounds (non-detected nodes)
        let tangleMinX = Infinity;
        otherNodes.forEach(node => {
            const x = node.x || 0;
            if (x < tangleMinX) tangleMinX = x;
        });
        
        // If no other nodes, use center as reference
        if (tangleMinX === Infinity) {
            tangleMinX = width * 0.5;
        }
        
        // Position detected nodes to the LEFT of tangle - CLEAR of all text
        // Find the furthest node to measure canvas extent
        let maxNodeX = 0;
        nodes.forEach(node => {
            const x = node.x || 0;
            if (x > maxNodeX) maxNodeX = x;
        });
        
        // Use 20% of the max canvas spread as our target distance from tangle
        // Reduced offset for tighter clustering near main graph
        const canvasSpread = maxNodeX - tangleMinX;
        const targetDistance = Math.max(120, canvasSpread * 0.20) + 40;
        const targetX = tangleMinX - targetDistance;
        
        console.log("tangleMinX:", tangleMinX.toFixed(0), "targetDistance:", targetDistance.toFixed(0), "targetX:", targetX.toFixed(0));
        
        console.log("Canvas spread:", canvasSpread.toFixed(0), "targetX:", targetX.toFixed(0));
        
        // Vertical layout - centered, with spacing
        const spacing = Math.min(45, (height * 0.7) / Math.max(1, detectedNodes.length));
        const totalHeight = detectedNodes.length * spacing;
        const startY = (height - totalHeight) / 2;
        
        console.log(`Tangle left edge: ${tangleMinX.toFixed(0)}, placing detected at X=${targetX.toFixed(0)}`);
        
        // Calculate target positions
        detectedNodes.forEach((node, i) => {
            node.fx = targetX + (Math.random() - 0.5) * 20;
            node.fy = startY + i * spacing;
        });
        
        // Use D3 transition on the actual DOM elements
        if (this.graph.nodesGroup) {
            this.graph.nodesGroup.selectAll('g.node')
                .filter(d => detectedIds.has(d.id))
                .transition()
                .duration(800)
                .ease(d3.easeCubicOut)
                .attr('transform', d => `translate(${d.fx}, ${d.fy})`)
                .on('end', () => {
                    // Update the data position to match
                    detectedNodes.forEach(node => {
                        node.x = node.fx;
                        node.y = node.fy;
                    });
                });
            
            // Update links during transition
            const updateLinks = () => {
                if (this.graph.linksGroup) {
                    this.graph.linksGroup.selectAll('line')
                        .attr('x1', d => d?.source?.fx ?? d?.source?.x ?? 0)
                        .attr('y1', d => d?.source?.fy ?? d?.source?.y ?? 0)
                        .attr('x2', d => d?.target?.fx ?? d?.target?.x ?? 0)
                        .attr('y2', d => d?.target?.fy ?? d?.target?.y ?? 0);
                }
            };
            
            // Update links periodically during animation
            const linkInterval = setInterval(updateLinks, 50);
            setTimeout(() => {
                clearInterval(linkInterval);
                updateLinks();
                this.reapplyHighlights();
                console.log('Clustering animation complete');
            }, 850);
        }
    }
    
    /**
     * Remove clustering force
     */
    removeClusteringForce() {
        // Unpin all nodes (remove fixed positions)
        const nodes = this.graph.data?.nodes || [];
        nodes.forEach(node => {
            node.fx = null;
            node.fy = null;
        });
        
        // Restart simulation
        if (this.graph.simulation) {
            this.graph.simulation.simulation.alpha(0.5).restart();
        }
        
        this.clusterForce = null;
        console.log('Clustering removed - nodes unpinned');
        
        // Re-apply highlights
        setTimeout(() => this.reapplyHighlights(), 300);
    }
    

    // ========================================================================
    // NARRATIVE CLUSTERING (U7) - Opposition swarm layout
    // ========================================================================
    
    /**
     * Classify claims as pro/anti/neutral based on relationships and content
     */
    classifyNarrativeStances() {
        this.narrativeStances.clear();
        const nodes = this.graph.data?.nodes || [];
        const links = this.graph.data?.links || [];
        
        // Keywords indicating anti-narrative (dismissal/attack language)
        const antiKeywords = [
            'crazy', 'insane', 'delusional', 'mentally', 'unstable',
            'discredited', 'debunked', 'conspiracy', 'paranoid', 'radical',
            'extremist', 'fringe', 'crackpot', 'lunatic', 'unhinged',
            'false', 'lie', 'lied', 'hoax', 'myth', 'fabricat', 'damned lie',
            'denied', 'denials', 'chorus of denials'
        ];
        
        // Keywords indicating pro-narrative (supportive/validating language)
        const proKeywords = [
            'testified', 'exposed', 'revealed', 'warned', 'documented',
            'proved', 'confirmed', 'evidence', 'witness', 'truth',
            'hero', 'patriot', 'whistleblower', 'brave', 'courageous',
            'persecution', 'suppressed', 'silenced', 'censored',
            'blew the whistle', 'turned down', 'refused', 'rejected'
        ];
        
        nodes.forEach(node => {
            let stance = 'neutral';
            const text = (node.claim_text || node.text || node.label || node.name || '').toLowerCase();
            
            // Check edge relationships first (strongest signal)
            const nodeLinks = links.filter(l => 
                (l.source?.id || l.source) === node.id || 
                (l.target?.id || l.target) === node.id
            );
            
            for (const link of nodeLinks) {
                const relType = (link.type || link.relationship_type || '').toUpperCase();
                if (relType === 'CONTRADICTS' || relType === 'OPPOSES' || relType === 'DENIES') {
                    stance = 'anti';
                    break;
                }
                if (relType === 'SUPPORTS' || relType === 'CONFIRMS' || relType === 'VALIDATES') {
                    stance = 'pro';
                    break;
                }
            }
            
            // If no relationship signal, check keywords
            if (stance === 'neutral') {
                const antiCount = antiKeywords.filter(kw => text.includes(kw)).length;
                const proCount = proKeywords.filter(kw => text.includes(kw)).length;
                
                if (antiCount > proCount && antiCount >= 1) {
                    stance = 'anti';
                } else if (proCount > antiCount && proCount >= 1) {
                    stance = 'pro';
                }
            }
            
            // Check claim position field if available
            if (node.position === 'oppose' || node.position === 'opposing') {
                stance = 'anti';
            } else if (node.position === 'support' || node.position === 'supporting') {
                stance = 'pro';
            }
            
            this.narrativeStances.set(node.id, stance);
        });
        
        // Log stance distribution
        const stances = Array.from(this.narrativeStances.values());
        const proCt = stances.filter(s => s === 'pro').length;
        const antiCt = stances.filter(s => s === 'anti').length;
        const neutralCt = stances.filter(s => s === 'neutral').length;
        console.log(`Narrative stances classified: ${proCt} pro, ${antiCt} anti, ${neutralCt} neutral`);
    }
    
    /**
     * Apply narrative clustering force - pro left, anti right, neutral center
     */
    applyNarrativeClusteringForce() {
        const width = this.graph.width || 800;
        const height = this.graph.height || 600;
        const nodes = this.graph.data?.nodes || [];
        
        if (nodes.length === 0) {
            console.warn('GraphFilter: No nodes for narrative clustering');
            return;
        }
        
        // First, release ALL fixed positions from any previous clustering
        nodes.forEach(node => {
            node.fx = null;
            node.fy = null;
        });
        
        // Stop simulation
        if (this.graph.simulation?.simulation) {
            this.graph.simulation.simulation.stop();
        } else if (this.graph.simulation) {
            this.graph.simulation.stop();
        }
        
        // Classify all nodes
        this.classifyNarrativeStances();
        
        // Separate by stance
        const proNodes = nodes.filter(n => this.narrativeStances.get(n.id) === 'pro');
        const antiNodes = nodes.filter(n => this.narrativeStances.get(n.id) === 'anti');
        const neutralNodes = nodes.filter(n => this.narrativeStances.get(n.id) === 'neutral');
        
        console.log(`Positioning: ${proNodes.length} pro (left), ${antiNodes.length} anti (right), ${neutralNodes.length} neutral (center)`);
        
        // === ORGANIC HONEYCOMB LAYOUT ===
        // Stagger rows so text labels don't overlap
        // Each row offset horizontally from previous
        
        const nodeRadius = 12;  // Approximate node size
        const textWidth = 180;  // Space needed for text label
        const verticalGap = 55; // Vertical space between rows
        const horizontalStagger = textWidth * 0.6;  // Offset for each row
        
        // PRO nodes (far left, staggered column going down-right)
        const proBaseX = 30;  // Start near left edge
        proNodes.forEach((node, i) => {
            const row = i;
            const stagger = (row % 3) * horizontalStagger;  // 3-phase stagger
            node.fx = proBaseX + stagger + (Math.random() - 0.5) * 20;
            node.fy = 80 + row * verticalGap + (Math.random() - 0.5) * 10;
        });
        
        // ANTI nodes (far right, staggered column going down-left)
        const antiBaseX = width - 30;  // Start near right edge
        antiNodes.forEach((node, i) => {
            const row = i;
            const stagger = (row % 3) * horizontalStagger;  // 3-phase stagger going left
            node.fx = antiBaseX - stagger + (Math.random() - 0.5) * 20;
            node.fy = 80 + row * verticalGap + (Math.random() - 0.5) * 10;
        });
        
        // NEUTRAL nodes - organic honeycomb filling the center
        // Use hexagonal packing with text-aware spacing
        const neutralZoneLeft = width * 0.18;
        const neutralZoneRight = width * 0.82;
        const neutralZoneWidth = neutralZoneRight - neutralZoneLeft;
        const neutralZoneTop = 70;
        const neutralZoneBottom = height - 50;
        const neutralZoneHeight = neutralZoneBottom - neutralZoneTop;
        
        // Calculate optimal columns based on text width needs
        const effectiveNodeWidth = textWidth * 0.7;  // Overlap allowed
        const cols = Math.max(3, Math.floor(neutralZoneWidth / effectiveNodeWidth));
        const rows = Math.ceil(neutralNodes.length / cols);
        
        // Spacing
        const colSpacing = neutralZoneWidth / cols;
        const rowSpacing = Math.max(verticalGap, neutralZoneHeight / rows);
        
        neutralNodes.forEach((node, i) => {
            const col = i % cols;
            const row = Math.floor(i / cols);
            
            // Honeycomb offset: odd rows shift right by half column width
            const honeycombOffset = (row % 2) * (colSpacing * 0.5);
            
            // Additional micro-stagger within each cell for organic feel
            const microOffsetX = (Math.random() - 0.5) * 25;
            const microOffsetY = (Math.random() - 0.5) * 15;
            
            node.fx = neutralZoneLeft + col * colSpacing + honeycombOffset + colSpacing * 0.5 + microOffsetX;
            node.fy = neutralZoneTop + row * rowSpacing + rowSpacing * 0.5 + microOffsetY;
        });
        
        // Animate all nodes to new positions
        if (this.graph.nodesGroup) {
            this.graph.nodesGroup.selectAll('g.node')
                .transition()
                .duration(800)
                .ease(d3.easeCubicOut)
                .attr('transform', d => `translate(${d.fx}, ${d.fy})`)
                .on('end', function(d) {
                    d.x = d.fx;
                    d.y = d.fy;
                });
            
            // Update links during transition
            const updateLinks = () => {
                if (this.graph.linksGroup) {
                    this.graph.linksGroup.selectAll('line')
                        .attr('x1', d => d?.source?.fx ?? d?.source?.x ?? 0)
                        .attr('y1', d => d?.source?.fy ?? d?.source?.y ?? 0)
                        .attr('x2', d => d?.target?.fx ?? d?.target?.x ?? 0)
                        .attr('y2', d => d?.target?.fy ?? d?.target?.y ?? 0);
                }
            };
            
            const linkInterval = setInterval(updateLinks, 50);
            setTimeout(() => {
                clearInterval(linkInterval);
                updateLinks();
                this.renderNarrativeLabels();
                console.log('Narrative clustering complete');
            }, 850);
        }
    }
    
    /**
     * Render labels for narrative poles
     */
    renderNarrativeLabels() {
        const g = this.graph.g;
        const width = this.graph.width || 800;
        
        // Remove existing labels
        g.selectAll('.narrative-label').remove();
        
        // Count nodes in each category for the labels
        const stances = Array.from(this.narrativeStances.values());
        const proCt = stances.filter(s => s === 'pro').length;
        const antiCt = stances.filter(s => s === 'anti').length;
        const neutralCt = stances.filter(s => s === 'neutral').length;
        
        // Add pole labels with counts - positioned for organic staggered layout
        g.append('text')
            .attr('class', 'narrative-label')
            .attr('x', 30)
            .attr('y', 50)
            .attr('text-anchor', 'start')
            .attr('fill', '#4CAF50')
            .attr('font-size', '14px')
            .attr('font-weight', 'bold')
            .text(`↙ PRO (${proCt})`);
        
        g.append('text')
            .attr('class', 'narrative-label')
            .attr('x', width - 30)
            .attr('y', 50)
            .attr('text-anchor', 'end')
            .attr('fill', '#E91E63')
            .attr('font-size', '14px')
            .attr('font-weight', 'bold')
            .text(`ANTI (${antiCt}) ↘`);
        
        g.append('text')
            .attr('class', 'narrative-label')
            .attr('x', width * 0.5)
            .attr('y', 50)
            .attr('text-anchor', 'middle')
            .attr('fill', '#9E9E9E')
            .attr('font-size', '12px')
            .text(`NEUTRAL (${neutralCt})`);
    }
    
    /**
     * Remove narrative clustering force
     */
    removeNarrativeClusteringForce() {
        // Remove labels
        if (this.graph.g) {
            this.graph.g.selectAll('.narrative-label').remove();
        }
        
        // Unpin all nodes
        const nodes = this.graph.data?.nodes || [];
        nodes.forEach(node => {
            node.fx = null;
            node.fy = null;
        });
        
        // Clear stance data
        this.narrativeStances.clear();
        
        // Restart simulation
        if (this.graph.simulation?.simulation) {
            this.graph.simulation.simulation.alpha(0.5).restart();
        } else if (this.graph.simulation?.restart) {
            this.graph.simulation.restart(0.5);
        }
        
        console.log('Narrative clustering removed');
    }


    /**
     * Get cluster visualization data
     */
    getClusterData() {
        if (!this.clusteringEnabled || this.detectedNodeIds.size === 0) {
            return null;
        }
        
        const detectedNodes = this.graph.data?.nodes?.filter(n => n.isDetected) || [];
        
        if (detectedNodes.length === 0) return null;
        
        // Calculate cluster bounds
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        
        detectedNodes.forEach(node => {
            minX = Math.min(minX, node.x || 0);
            maxX = Math.max(maxX, node.x || 0);
            minY = Math.min(minY, node.y || 0);
            maxY = Math.max(maxY, node.y || 0);
        });
        
        const padding = 30;
        
        return {
            x: minX - padding,
            y: minY - padding,
            width: (maxX - minX) + padding * 2,
            height: (maxY - minY) + padding * 2,
            nodeCount: detectedNodes.length,
            centerX: (minX + maxX) / 2,
            centerY: (minY + maxY) / 2
        };
    }
    
    /**
     * Render cluster boundary visualization
     */
    renderClusterBoundary(svg) {
        // Remove existing boundary
        svg.select('.cluster-boundary').remove();
        
        if (!this.clusteringEnabled) return;
        
        const clusterData = this.getClusterData();
        if (!clusterData) return;
        
        const boundary = svg.insert('g', ':first-child')
            .attr('class', 'cluster-boundary');
        
        // Draw boundary ellipse
        boundary.append('ellipse')
            .attr('cx', clusterData.centerX)
            .attr('cy', clusterData.centerY)
            .attr('rx', clusterData.width / 2)
            .attr('ry', clusterData.height / 2)
            .attr('fill', 'rgba(255, 56, 96, 0.1)')
            .attr('stroke', CONFIG.COLORS.SUPPRESSION_HIGH || '#FF3860')
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', '8,4');
        
        // Add label
        boundary.append('text')
            .attr('x', clusterData.centerX)
            .attr('y', clusterData.y - 10)
            .attr('text-anchor', 'middle')
            .attr('fill', CONFIG.COLORS.SUPPRESSION_HIGH || '#FF3860')
            .attr('font-size', '12px')
            .attr('font-weight', 'bold')
            .text(`Detected Cluster (${clusterData.nodeCount} nodes)`);
    }
    
    // ========================================================================
    // PUBLIC API
    // ========================================================================
    
    /**
     * Show/hide filter panel
     */
    toggle() {
        this.isCollapsed = !this.isCollapsed;
        if (this.panel) {
            this.panel.classList.toggle('collapsed', this.isCollapsed);
            this.panel.querySelector('.gf-collapse-icon').textContent = this.isCollapsed ? '▶' : '▼';
        }
    }
    
    /**
     * Programmatically set max nodes
     */
    setMaxNodes(count) {
        this.filters.maxNodes = count;
        if (this.panel) {
            this.panel.querySelector('#gf-node-count').value = count;
            this.panel.querySelector('#gf-node-count-value').textContent = count;
        }
        this.applyFilters();
    }
    
    /**
     * Destroy and cleanup
     */
    destroy() {
        this.removeClusteringForce();
        if (this.panel) {
            this.panel.remove();
            this.panel = null;
        }
    }
}

// ============================================================================
// CSS STYLES (injected once)
// ============================================================================

const FILTER_STYLES = `
.graph-filter-panel {
    position: absolute;
    top: 240px;
    right: 10px;
    width: 220px;
    max-height: calc(100vh - 280px);
    background: rgba(22, 27, 34, 0.95);
    border: 1px solid #30363d;
    border-radius: 8px;
    font-size: 12px;
    z-index: 100;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    transition: width 0.2s;
    display: flex;
    flex-direction: column;
}

.graph-filter-panel.collapsed {
    width: auto;
}

.graph-filter-panel.collapsed .gf-body {
    display: none;
}

.gf-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    cursor: pointer;
    border-bottom: 1px solid #30363d;
    user-select: none;
}

.gf-header:hover {
    background: rgba(255, 255, 255, 0.05);
}

.gf-icon {
    font-size: 14px;
}

.gf-title {
    flex: 1;
    font-weight: 600;
    color: #e6edf3;
}

.gf-collapse-icon {
    color: #8b949e;
    font-size: 10px;
}

.gf-body {
    padding: 12px;
    overflow-y: scroll;
    max-height: 300px;
    overflow-x: hidden;
}

.gf-body::-webkit-scrollbar {
    width: 6px;
}

.gf-body::-webkit-scrollbar-track {
    background: #0d1117;
    border-radius: 3px;
}

.gf-body::-webkit-scrollbar-thumb {
    background: #30363d;
    border-radius: 3px;
}

.gf-body::-webkit-scrollbar-thumb:hover {
    background: #484f58;
}

.gf-group {
    margin-bottom: 14px;
}

.gf-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    color: #8b949e;
    font-size: 11px;
    text-transform: uppercase;
}

.gf-value {
    color: #58a6ff;
    font-weight: 600;
}

.gf-slider {
    width: 100%;
    height: 4px;
    border-radius: 2px;
    background: #30363d;
    outline: none;
    -webkit-appearance: none;
}

.gf-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #58a6ff;
    cursor: pointer;
    border: 2px solid #0d1117;
}

.gf-slider::-webkit-slider-thumb:hover {
    background: #79b8ff;
}

.gf-toggles {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.gf-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    color: #e6edf3;
    font-size: 11px;
}

.gf-toggle input {
    display: none;
}

.gf-toggle-dot {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    opacity: 0.3;
    transition: opacity 0.15s;
}

.gf-toggle input:checked + .gf-toggle-dot {
    opacity: 1;
}

.gf-toggle-highlight {
    padding: 6px 8px;
    background: rgba(88, 166, 255, 0.1);
    border-radius: 4px;
    border: 1px solid transparent;
}

.gf-toggle-highlight:has(input:checked) {
    border-color: #58a6ff;
    background: rgba(88, 166, 255, 0.2);
}

.gf-claim-types {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px;
}

.gf-badge {
    background: #f85149;
    color: white;
    padding: 1px 6px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 600;
}

.gf-stats {
    padding: 8px;
    background: #0d1117;
    border-radius: 4px;
    text-align: center;
    color: #8b949e;
    font-size: 10px;
    margin-bottom: 10px;
}

.gf-reset-btn {
    width: 100%;
    padding: 8px;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 4px;
    color: #e6edf3;
    font-size: 11px;
    cursor: pointer;
    transition: background 0.15s;
}

.gf-reset-btn:hover {
    background: #30363d;
}

/* Cluster boundary animation */
.cluster-boundary ellipse {
    animation: cluster-pulse 2s ease-in-out infinite;
}

@keyframes cluster-pulse {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
}
`;

// Inject styles once
if (!document.getElementById('graph-filter-styles')) {
    const style = document.createElement('style');
    style.id = 'graph-filter-styles';
    style.textContent = FILTER_STYLES;
    document.head.appendChild(style);
}

// Export for use
export default GraphFilter;
