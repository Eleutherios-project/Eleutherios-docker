/**
 * Aegis Insight v2.0 - Detection Highlighter
 * Applies visual highlighting based on detection mode and results
 */

import { CONFIG, getScoreColor } from '../config.js';
// Note: If this file is in js/graph/, the import path '../config.js' reaches js/config.js

export class DetectionHighlighter {
    constructor(graphRenderer) {
        this.graph = graphRenderer;
        this.activeMode = null;
        this.detectionResults = null;
        this.affectedNodeIds = new Set();
        this.affectedEdgeIds = new Set();
    }

    /**
     * Apply detection highlighting to the graph
     * @param {Object} results - Detection results from API
     * @param {string} mode - 'suppression', 'coordination', or 'anomaly'
     */
    applyHighlights(results, mode) {
        // Clear previous highlights first
        this.clearHighlights();
        this.activeMode = mode;
        this.detectionResults = results;
        
        // Build affected sets
        this.buildAffectedSets(results, mode);
        
        // Apply mode-specific highlighting
        switch (mode) {
            case 'suppression':
                this.applySuppressionHighlights(results);
                break;
            case 'coordination':
                this.applyCoordinationHighlights(results);
                break;
            case 'anomaly':
                this.applyAnomalyHighlights(results);
                break;
        }
        
        // Add legend
        this.showLegend(mode);
    }

    /**
     * Build sets of affected node and edge IDs
     * Handles various ID formats and matching strategies
     */
    buildAffectedSets(results, mode) {
        this.affectedNodeIds.clear();
        this.affectedEdgeIds.clear();

        // Collect all claim IDs from various result fields
        const claimIds = new Set();
        
        // From two-stage pipeline (frontend attaches these)
        if (results.claim_ids && Array.isArray(results.claim_ids)) {
            results.claim_ids.forEach(id => claimIds.add(id));
        }

        // From backend analyzed_claims
        if (results.analyzed_claims) {
            results.analyzed_claims.forEach(c => {
                if (c.graph_id) claimIds.add(c.graph_id);  // D3 graph ID
                if (c.claim_id) claimIds.add(c.claim_id);
                if (c.id) claimIds.add(c.id);
            });
        }

        // From suppression detector affected_claims
        if (results.affected_claims) {
            results.affected_claims.forEach(c => {
                if (c.graph_id) claimIds.add(c.graph_id);  // D3 graph ID - matches "claim-{elementId}"
                if (c.claim_id) claimIds.add(c.claim_id);
                if (c.id) claimIds.add(c.id);
            });
        }

        // From suppressed_claims (legacy format)
        if (results.suppressed_claims) {
            results.suppressed_claims.forEach(id => claimIds.add(id));
        }

        // From coordination detector coordinated_claims
        if (results.coordinated_claims) {
            results.coordinated_claims.forEach(c => {
                if (c.graph_id) claimIds.add(c.graph_id);
                if (c.claim_id) claimIds.add(c.claim_id);
                if (c.id) claimIds.add(c.id);
            });
        }

        // From anomaly detector anomalous_claims
        if (results.anomalous_claims) {
            results.anomalous_claims.forEach(c => {
                if (c.graph_id) claimIds.add(c.graph_id);
                if (c.claim_id) claimIds.add(c.claim_id);
                if (c.id) claimIds.add(c.id);
            });
        }

        // Add all collected IDs to the affected set
        claimIds.forEach(id => this.affectedNodeIds.add(id));

        console.log('Detection highlighting: tracking', this.affectedNodeIds.size, 'affected node IDs:', [...this.affectedNodeIds].slice(0, 5));

        // Match claim IDs to graph nodes
        // Graph node IDs look like: "claim-4:uuid:12345" 
        // Claim IDs from API might be: "claim-4:uuid:12345", "12345", or just the numeric part
        if (this.graph.data?.nodes) {
            this.graph.data.nodes.forEach(node => {
                // Check if any claim ID matches this node
                for (const claimId of claimIds) {
                    if (this.nodeMatchesClaimId(node.id, claimId)) {
                        this.affectedNodeIds.add(node.id);
                        break;
                    }
                }
            });
        }
        
        console.log(`Detection: Matched ${this.affectedNodeIds.size} graph nodes`);
    }
    
    /**
     * Check if a graph node ID matches a claim ID
     * Handles various ID formats flexibly
     * 
     * Graph nodes: "claim-4:uuid:12345"
     * Pattern search: "4:uuid:12345"
     */
    nodeMatchesClaimId(nodeId, claimId) {
        if (!nodeId || !claimId) return false;
        
        // Normalize both IDs by removing "claim-" or "entity-" prefix if present
        const normalizeId = (id) => {
            let normalized = String(id);
            if (normalized.startsWith('claim-')) {
                normalized = normalized.substring(6);  // Remove "claim-"
            } else if (normalized.startsWith('entity-')) {
                normalized = normalized.substring(7);  // Remove "entity-"
            }
            return normalized;
        };
        
        const normalizedNodeId = normalizeId(nodeId);
        const normalizedClaimId = normalizeId(claimId);
        
        // Exact match after normalization
        if (normalizedNodeId === normalizedClaimId) return true;
        
        // Compare last segments (numeric IDs) as fallback
        const nodeNumericId = normalizedNodeId.split(':').pop();
        const claimNumericId = normalizedClaimId.split(':').pop();
        
        if (nodeNumericId === claimNumericId) return true;
        
        return false;
    }

    /**
     * Apply suppression mode highlighting
     * - Red borders on affected claims
     * - Dim META claims
     * - Brighten PRIMARY claims
     */
    applySuppressionHighlights(results) {
        const svg = this.graph.svg;
        const nodesGroup = this.graph.nodesGroup;
        const linksGroup = this.graph.linksGroup;
        const self = this;  // Reference for inner function
        
        // Ensure glow filter exists
        this.ensureGlowFilter(svg, 'suppression-glow', CONFIG.COLORS.SUPPRESSION_HIGH);
        
        // Update node visuals
        nodesGroup.selectAll('g.node').each(function(d) {
            const node = d3.select(this);
            const isAffected = self.affectedNodeIds.has(d.id);
            
            // Get claim type
            const claimType = d.claim_type || d.type || '';
            
            // Apply highlighting to affected claims
            if (isAffected) {
                console.log('HIGHLIGHTING NODE:', d.id, d.label?.substring(0, 40));
                // Red border with glow
                node.select('circle, rect, path')
                    .attr('stroke', CONFIG.COLORS.SUPPRESSION_HIGH)
                    .attr('stroke-width', CONFIG.SIZES.STROKE_WIDTH_DETECTED || 3)
                    .attr('stroke-dasharray', CONFIG.SUPPRESSION?.BORDER_DASH || '5,5')
                    .attr('filter', 'url(#suppression-glow)');
                
                // Add suppression indicator dot
                if (!node.select('.suppression-indicator').size()) {
                    node.append('circle')
                        .attr('class', 'suppression-indicator detection-highlight')
                        .attr('r', 4)
                        .attr('cx', (CONFIG.SIZES?.NODE_RADIUS_SMALL || 8) + 2)
                        .attr('cy', -(CONFIG.SIZES?.NODE_RADIUS_SMALL || 8) - 2)
                        .attr('fill', CONFIG.COLORS.SUPPRESSION_HIGH);
                }
            }
            
            // Dim META claims (they're the "dismissals")
            if (claimType.toUpperCase() === 'META') {
                node.style('opacity', CONFIG.SUPPRESSION?.META_OPACITY || 0.4);
            }
            
            // Brighten PRIMARY claims
            if (claimType.toUpperCase() === 'PRIMARY' && isAffected) {
                node.select('circle, rect, path')
                    .attr('stroke', '#4CAF50')  // Green for primary
                    .attr('stroke-width', 3);
            }
        });
    }
    
    /**
     * Ensure glow filter exists in SVG defs
     */
    ensureGlowFilter(svg, filterId, color) {
        let defs = svg.select('defs');
        if (defs.empty()) {
            defs = svg.append('defs');
        }
        
        if (defs.select(`#${filterId}`).empty()) {
            const filter = defs.append('filter')
                .attr('id', filterId)
                .attr('x', '-50%')
                .attr('y', '-50%')
                .attr('width', '200%')
                .attr('height', '200%');
            
            filter.append('feGaussianBlur')
                .attr('in', 'SourceGraphic')
                .attr('stdDeviation', '4')
                .attr('result', 'blur');
            
            filter.append('feFlood')
                .attr('flood-color', color)
                .attr('flood-opacity', '0.6')
                .attr('result', 'color');
            
            filter.append('feComposite')
                .attr('in', 'color')
                .attr('in2', 'blur')
                .attr('operator', 'in')
                .attr('result', 'glow');
            
            const merge = filter.append('feMerge');
            merge.append('feMergeNode').attr('in', 'glow');
            merge.append('feMergeNode').attr('in', 'SourceGraphic');
        }
    }

    /**
     * Apply coordination mode highlighting
     * - Orange clusters around coordinated groups
     * - Thick orange lines between coordinated claims
     * - Temporal labels
     * - Pulse animation on synchronized claims
     */
    applyCoordinationHighlights(results) {
        const svg = this.graph.svg;
        const nodesGroup = this.graph.nodesGroup;
        const linksGroup = this.graph.linksGroup;
        const g = this.graph.g;
        
        // Create cluster overlay group
        let clusterGroup = g.select('.coordination-clusters');
        if (clusterGroup.empty()) {
            clusterGroup = g.insert('g', '.links')
                .attr('class', 'coordination-clusters');
        }
        
        // TEMPORARILY DISABLED - cluster ellipse needs positioning fix
        // Draw cluster backgrounds
        if (false && results.clusters) { // Disabled
            results.clusters.forEach((cluster, i) => {
                if (!cluster.claim_ids || cluster.claim_ids.length < 2) return;
                
                // Find cluster nodes
                const clusterNodes = this.graph.data.nodes.filter(n => 
                    cluster.claim_ids.includes(n.id));
                
                if (clusterNodes.length < 2) return;
                
                // Calculate cluster bounds
                const bounds = this.calculateClusterBounds(clusterNodes);
                
                // Draw cluster ellipse
                clusterGroup.append('ellipse')
                    .attr('class', 'cluster-bg')
                    .attr('cx', bounds.centerX)
                    .attr('cy', bounds.centerY)
                    .attr('rx', bounds.radiusX + CONFIG.COORDINATION.CLUSTER_PADDING)
                    .attr('ry', bounds.radiusY + CONFIG.COORDINATION.CLUSTER_PADDING)
                    .attr('fill', CONFIG.COLORS.COORDINATION_CLUSTER)
                    .attr('stroke', CONFIG.COLORS.COORDINATION_PRIMARY)
                    .attr('stroke-width', 2)
                    .attr('stroke-dasharray', '4,4');
                
                // Add cluster label
                clusterGroup.append('text')
                    .attr('class', 'cluster-label')
                    .attr('x', bounds.centerX)
                    .attr('y', bounds.centerY - bounds.radiusY - 25)
                    .attr('text-anchor', 'middle')
                    .attr('fill', CONFIG.COLORS.COORDINATION_PRIMARY)
                    .attr('font-size', '12px')
                    .attr('font-weight', 'bold')
                    .text(`Cluster ${i + 1}: ${cluster.claim_ids.length} claims`);
                
                // Add temporal range label if available
                if (cluster.temporal_range) {
                    clusterGroup.append('text')
                        .attr('class', 'cluster-temporal')
                        .attr('x', bounds.centerX)
                        .attr('y', bounds.centerY - bounds.radiusY - 10)
                        .attr('text-anchor', 'middle')
                        .attr('fill', CONFIG.COLORS.TEXT_SECONDARY)
                        .attr('font-size', '10px')
                        .text(cluster.temporal_range);
                }
            });
        }
        
        // Update node visuals
        const self = this;
        nodesGroup.selectAll('g.node').each(function(d) {
            const node = d3.select(this);
            const isCoordinated = self.affectedNodeIds.has(d.id); 
            
            if (isCoordinated) {
                // Orange border
                node.select('circle, rect, path')
                    .attr('stroke', CONFIG.COLORS.COORDINATION_PRIMARY)
                    .attr('stroke-width', CONFIG.SIZES.STROKE_WIDTH_DETECTED);
                
                // Add pulse animation
                const shape = node.select('circle, rect, path');
                const originalR = shape.attr('r') || CONFIG.SIZES.NODE_RADIUS_SMALL;
                
                // Pulse ring
                if (!node.select('.pulse-ring').size()) {
                    node.insert('circle', ':first-child')
                        .attr('class', 'pulse-ring')
                        .attr('r', originalR)
                        .attr('fill', 'none')
                        .attr('stroke', CONFIG.COLORS.COORDINATION_PRIMARY)
                        .attr('stroke-width', 2)
                        .attr('opacity', 0);
                }
                
                // Animate pulse
                const pulseRing = node.select('.pulse-ring');
                function pulse() {
                    pulseRing
                        .attr('r', originalR)
                        .attr('opacity', 0.8)
                        .transition()
                        .duration(CONFIG.COORDINATION.PULSE_DURATION)
                        .ease(d3.easeQuadOut)
                        .attr('r', parseFloat(originalR) * 2.5)
                        .attr('opacity', 0)
                        .on('end', pulse);
                }
                pulse();
                
                // Add publication date label if available
                const claimData = results.coordinated_claims?.find(c => (c.id || c.claim_id) === d.id); 
                if (claimData?.publication_date) {
                    node.append('text')
                        .attr('class', 'date-label')
                        .attr('dy', -CONFIG.SIZES.NODE_RADIUS_SMALL - 15)
                        .attr('text-anchor', 'middle')
                        .attr('fill', CONFIG.COLORS.COORDINATION_PRIMARY)
                        .attr('font-size', '9px')
                        .text(claimData.publication_date);
                }
            }
        });
        
        // Draw coordination lines between synchronized claims
        if (results.coordinated_pairs) {
            results.coordinated_pairs.forEach(pair => {
                const sourceNode = this.graph.data.nodes.find(n => n.id === pair.source_id);
                const targetNode = this.graph.data.nodes.find(n => n.id === pair.target_id);
                
                if (sourceNode && targetNode) {
                    linksGroup.append('line')
                        .attr('class', 'coordination-link')
                        .attr('x1', sourceNode.x)
                        .attr('y1', sourceNode.y)
                        .attr('x2', targetNode.x)
                        .attr('y2', targetNode.y)
                        .attr('stroke', CONFIG.COLORS.COORDINATION_PRIMARY)
                        .attr('stroke-width', CONFIG.SIZES.EDGE_WIDTH_THICK)
                        .attr('stroke-opacity', 0.8);
                }
            });
        }
    }

    /**
     * Apply anomaly mode highlighting
     * - Purple connections between anomalous patterns
     * - Distance labels
     * - Cultural/geographic tags
     */
    applyAnomalyHighlights(results) {
        const svg = this.graph.svg;
        const nodesGroup = this.graph.nodesGroup;
        const linksGroup = this.graph.linksGroup;
        const g = this.graph.g;
        
        // Create anomaly overlay group
        let anomalyGroup = g.select('.anomaly-connections');
        if (anomalyGroup.empty()) {
            anomalyGroup = g.append('g')
                .attr('class', 'anomaly-connections');
        }
        
        // Process anomalous patterns
        if (results.anomalous_patterns) {
            results.anomalous_patterns.forEach((pattern, i) => {
                // Highlight claims in this pattern
                if (pattern.claims) {
                    pattern.claims.forEach(claim => {
                        const node = nodesGroup.select(`g.node`)
                            .filter(d => d.id === (claim.id || claim.claim_id));
                        
                        if (!node.empty()) {
                            // Purple border
                            node.select('circle, rect, path')
                                .attr('stroke', CONFIG.COLORS.ANOMALY_PRIMARY)
                                .attr('stroke-width', CONFIG.SIZES.STROKE_WIDTH_DETECTED);
                            
                            // Add cultural/geographic tag
                            if (claim.culture || claim.region) {
                                node.append('text')
                                    .attr('class', 'culture-tag')
                                    .attr('dy', -CONFIG.SIZES.NODE_RADIUS_SMALL - 15)
                                    .attr('text-anchor', 'middle')
                                    .attr('fill', CONFIG.COLORS.ANOMALY_CONNECTION)
                                    .attr('font-size', '9px')
                                    .attr('font-style', 'italic')
                                    .text(claim.culture || claim.region);
                            }
                        }
                    });
                }
                
                // Draw connections between pattern claims
                if (pattern.connections) {
                    pattern.connections.forEach(conn => {
                        const sourceNode = this.graph.data.nodes.find(n => n.id === conn.source_id);
                        const targetNode = this.graph.data.nodes.find(n => n.id === conn.target_id);
                        
                        if (sourceNode && targetNode) {
                            // Draw connection line
                            const line = anomalyGroup.append('line')
                                .attr('class', 'anomaly-link')
                                .attr('x1', sourceNode.x)
                                .attr('y1', sourceNode.y)
                                .attr('x2', targetNode.x)
                                .attr('y2', targetNode.y)
                                .attr('stroke', CONFIG.COLORS.ANOMALY_CONNECTION)
                                .attr('stroke-width', 2)
                                .attr('stroke-dasharray', CONFIG.ANOMALY.CONNECTION_DASHARRAY)
                                .attr('stroke-opacity', 0.8);
                            
                            // Add distance label if available
                            if (conn.distance_km) {
                                const midX = (sourceNode.x + targetNode.x) / 2;
                                const midY = (sourceNode.y + targetNode.y) / 2;
                                
                                anomalyGroup.append('text')
                                    .attr('class', 'distance-label')
                                    .attr('x', midX)
                                    .attr('y', midY - 8)
                                    .attr('text-anchor', 'middle')
                                    .attr('fill', CONFIG.COLORS.ANOMALY_PRIMARY)
                                    .attr('font-size', '10px')
                                    .attr('font-weight', 'bold')
                                    .text(`${conn.distance_km.toLocaleString()} km`);
                            }
                            
                            // Add temporal overlap label
                            if (conn.temporal_overlap) {
                                const midX = (sourceNode.x + targetNode.x) / 2;
                                const midY = (sourceNode.y + targetNode.y) / 2;
                                
                                anomalyGroup.append('text')
                                    .attr('class', 'temporal-label')
                                    .attr('x', midX)
                                    .attr('y', midY + 12)
                                    .attr('text-anchor', 'middle')
                                    .attr('fill', CONFIG.COLORS.TEXT_SECONDARY)
                                    .attr('font-size', '9px')
                                    .text(conn.temporal_overlap);
                            }
                        }
                    });
                }
            });
        }
    }

    /**
     * Calculate bounds for a cluster of nodes
     */
    calculateClusterBounds(nodes) {
        const xs = nodes.map(n => n.x);
        const ys = nodes.map(n => n.y);
        
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);
        
        return {
            centerX: (minX + maxX) / 2,
            centerY: (minY + maxY) / 2,
            radiusX: (maxX - minX) / 2 + 20,
            radiusY: (maxY - minY) / 2 + 20,
        };
    }

    /**
     * Ensure necessary SVG filters exist
     */
    ensureFilters() {
        const defs = this.graph.svg.select('defs');
        
        // Brightness boost filter
        if (defs.select('#brightness-boost').empty()) {
            const filter = defs.append('filter')
                .attr('id', 'brightness-boost');
            
            filter.append('feComponentTransfer')
                .append('feFuncR')
                .attr('type', 'linear')
                .attr('slope', CONFIG.SUPPRESSION.PRIMARY_BRIGHTNESS);
        }
    }

    /**
     * Show detection mode legend
     */
    showLegend(mode) {
        // Remove existing legend
        d3.select('#detection-legend').remove();
        
        const legends = {
            suppression: [
                { color: CONFIG.COLORS.SUPPRESSION_HIGH, label: 'Suppressed claim', shape: 'circle', dashed: true },
                { color: CONFIG.COLORS.SUPPRESSION_HIGH, label: 'Broken citation', shape: 'line', dashed: true },
                { color: 'rgba(255,255,255,0.4)', label: 'META claim (dimmed)', shape: 'circle' },
                { color: CONFIG.COLORS.TRUST_HIGH, label: 'PRIMARY claim', shape: 'circle' },
            ],
            coordination: [
                { color: CONFIG.COLORS.COORDINATION_PRIMARY, label: 'Coordinated claim', shape: 'circle' },
                { color: CONFIG.COLORS.COORDINATION_CLUSTER, label: 'Coordination cluster', shape: 'ellipse', stroke: CONFIG.COLORS.COORDINATION_PRIMARY },
                { color: CONFIG.COLORS.COORDINATION_PRIMARY, label: 'Sync connection', shape: 'line' },
            ],
            anomaly: [
                { color: CONFIG.COLORS.ANOMALY_PRIMARY, label: 'Anomalous pattern', shape: 'circle' },
                { color: CONFIG.COLORS.ANOMALY_CONNECTION, label: 'Cross-cultural link', shape: 'line', dashed: true },
            ],
        };
        
        const items = legends[mode];
        if (!items) return;
        
        const legend = d3.select(this.graph.container)
            .append('div')
            .attr('id', 'detection-legend')
            .attr('class', 'detection-legend')
            .style('position', 'absolute')
            .style('top', '20px')
            .style('right', '20px')
            .style('background', 'rgba(28, 36, 52, 0.95)')
            .style('border', '1px solid #2a3a5a')
            .style('border-radius', '8px')
            .style('padding', '12px 16px')
            .style('z-index', '100');
        
        legend.append('div')
            .style('font-weight', 'bold')
            .style('margin-bottom', '10px')
            .style('color', CONFIG.DETECTION_MODES[mode].color)
            .text(`${CONFIG.DETECTION_MODES[mode].icon} ${CONFIG.DETECTION_MODES[mode].name}`);
        
        items.forEach(item => {
            const row = legend.append('div')
                .style('display', 'flex')
                .style('align-items', 'center')
                .style('gap', '8px')
                .style('margin-bottom', '6px');
            
            // Draw symbol
            const svg = row.append('svg')
                .attr('width', 20)
                .attr('height', 16);
            
            if (item.shape === 'circle') {
                svg.append('circle')
                    .attr('cx', 10)
                    .attr('cy', 8)
                    .attr('r', 6)
                    .attr('fill', item.color)
                    .attr('stroke', item.dashed ? item.color : 'none')
                    .attr('stroke-dasharray', item.dashed ? '2,2' : 'none');
            } else if (item.shape === 'line') {
                svg.append('line')
                    .attr('x1', 2)
                    .attr('y1', 8)
                    .attr('x2', 18)
                    .attr('y2', 8)
                    .attr('stroke', item.color)
                    .attr('stroke-width', 2)
                    .attr('stroke-dasharray', item.dashed ? '4,2' : 'none');
            } else if (item.shape === 'ellipse') {
                svg.append('ellipse')
                    .attr('cx', 10)
                    .attr('cy', 8)
                    .attr('rx', 8)
                    .attr('ry', 5)
                    .attr('fill', item.color)
                    .attr('stroke', item.stroke || 'none')
                    .attr('stroke-width', 1);
            }
            
            row.append('span')
                .style('color', CONFIG.COLORS.TEXT_SECONDARY)
                .style('font-size', '12px')
                .text(item.label);
        });
        
        // Add clear button
        legend.append('button')
            .attr('class', 'clear-detection-btn')
            .style('margin-top', '12px')
            .style('width', '100%')
            .style('padding', '6px 12px')
            .style('background', 'rgba(255,255,255,0.1)')
            .style('border', '1px solid #2a3a5a')
            .style('border-radius', '4px')
            .style('color', CONFIG.COLORS.TEXT_PRIMARY)
            .style('cursor', 'pointer')
            .text('Clear Detection')
            .on('click', () => this.clearHighlights());
    }

    /**
     * Clear all detection highlights
     */
    clearHighlights() {
        this.activeMode = null;
        this.detectionResults = null;
        this.affectedNodeIds.clear();
        this.affectedEdgeIds.clear();
        
        const nodesGroup = this.graph.nodesGroup;
        const linksGroup = this.graph.linksGroup;
        const g = this.graph.g;
        
        // Reset node visuals
        nodesGroup.selectAll('g.node')
            .style('opacity', 1)
            .each(function(d) {
                const node = d3.select(this);
                
                // Remove detection indicators
                node.selectAll('.suppression-indicator, .pulse-ring, .date-label, .culture-tag').remove();
                
                // Reset stroke
                node.select('circle, rect, path')
                    .attr('stroke', '#ffffff')
                    .attr('stroke-width', CONFIG.SIZES.STROKE_WIDTH_NORMAL)
                    .attr('stroke-dasharray', 'none')
                    .attr('filter', 'none');
            });
        
        // Remove coordination overlays
        g.selectAll('.coordination-clusters, .anomaly-connections').remove();
        
        // Remove coordination links
        linksGroup.selectAll('.coordination-link').remove();
        
        // Reset edge visuals
        linksGroup.selectAll('line:not(.coordination-link)')
            .each(function(d) {
                const edge = d3.select(this);
                const type = d.type || d.relationship_type || 'MENTIONS';
                const style = CONFIG.EDGE_STYLES[type] || CONFIG.EDGE_STYLES.MENTIONS;
                
                edge.attr('stroke', style.color)
                    .attr('stroke-width', style.width)
                    .attr('stroke-dasharray', style.dasharray)
                    .attr('stroke-opacity', style.opacity);
            });
        
        // Remove legend
        d3.select('#detection-legend').remove();
    }

    /**
     * Get currently active detection mode
     */
    getActiveMode() {
        return this.activeMode;
    }

    /**
     * Check if detection is currently active
     */
    isActive() {
        return this.activeMode !== null;
    }

    /**
     * Update highlight positions (call on simulation tick)
     */
    updatePositions() {
        if (!this.isActive()) return;
        
        const g = this.graph.g;
        
        // Update cluster positions
        if (this.activeMode === 'coordination' && this.detectionResults?.clusters) {
            // Recalculate cluster bounds and update ellipses
            // This is expensive - only do if clusters are moving
        }
        
        // Update anomaly connection positions
        g.selectAll('.anomaly-link').each(function(d) {
            // Update line positions based on node positions
        });
    }
}
