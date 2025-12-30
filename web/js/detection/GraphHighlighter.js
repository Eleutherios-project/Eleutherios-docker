/**
 * Graph Highlighter
 * Adds visual indicators to graph based on detection results
 */

export class GraphHighlighter {
    constructor(graphRenderer) {
        this.graphRenderer = graphRenderer;
        this.currentHighlights = null;
        this.originalStyles = new Map();
    }
    
    /**
     * Apply detection highlights to graph
     * @param {object} results - Detection results
     * @param {string} mode - Detection mode
     */
    applyHighlights(results, mode) {
        if (!this.graphRenderer) {
            console.warn('GraphHighlighter: No graph renderer available');
            return;
        }
        
        // Clear previous highlights
        this.clearHighlights();
        
        // Store current highlights
        this.currentHighlights = { results, mode };
        
        // Apply mode-specific highlights
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
        
        // Trigger graph re-render
        this.graphRenderer.render?.();
    }
    
    /**
     * Apply suppression detection highlights
     */
    applySuppressionHighlights(results) {
        const suppressedClaims = results.suppressed_claims || [];
        const score = results.suppression_score || 0;
        
        // Highlight suppressed claims with red borders
        suppressedClaims.forEach(claimId => {
            this.highlightNode(claimId, {
                borderColor: '#ff4444',
                borderWidth: 3,
                borderStyle: 'solid',
                label: 'SUPPRESSED',
                labelColor: '#ff4444'
            });
        });
        
        // Highlight high-quality primary claims
        if (results.signals?.quality_visibility_gap?.high_quality_claims) {
            results.signals.quality_visibility_gap.high_quality_claims.forEach(claimId => {
                this.highlightNode(claimId, {
                    glow: '#4caf50',
                    brightness: 1.3,
                    label: 'HIGH QUALITY',
                    labelColor: '#4caf50'
                });
            });
        }
        
        // Dim META claims
        if (results.signals?.meta_density?.meta_claims) {
            results.signals.meta_density.meta_claims.forEach(claimId => {
                this.highlightNode(claimId, {
                    opacity: 0.5,
                    saturation: 0.3,
                    label: 'META',
                    labelColor: '#999'
                });
            });
        }
        
        // Highlight broken citation paths
        if (results.signals?.network_isolation?.isolated_nodes) {
            results.signals.network_isolation.isolated_nodes.forEach(nodeId => {
                this.highlightNode(nodeId, {
                    borderColor: '#ff4444',
                    borderWidth: 2,
                    borderStyle: 'dashed'
                });
            });
        }
    }
    
    /**
     * Apply coordination detection highlights
     */
    applyCoordinationHighlights(results) {
        // Highlight temporally clustered claims
        if (results.signals?.temporal_clustering?.clustered_claims) {
            const clusterColor = '#ff9800';
            results.signals.temporal_clustering.clustered_claims.forEach(claimId => {
                this.highlightNode(claimId, {
                    borderColor: clusterColor,
                    borderWidth: 2,
                    glow: clusterColor,
                    label: 'BURST',
                    labelColor: clusterColor
                });
            });
        }
        
        // Highlight citation cartels
        if (results.signals?.citation_cartel?.cartel_members) {
            results.signals.citation_cartel.cartel_members.forEach(nodeId => {
                this.highlightNode(nodeId, {
                    borderColor: '#ff5722',
                    borderWidth: 3,
                    borderStyle: 'solid',
                    label: 'CARTEL',
                    labelColor: '#ff5722'
                });
            });
            
            // Draw thick lines between cartel members
            this.highlightCartelConnections(
                results.signals.citation_cartel.cartel_members
            );
        }
        
        // Highlight centralized sources
        if (results.signals?.source_centralization?.top_source_claims) {
            results.signals.source_centralization.top_source_claims.forEach(claimId => {
                this.highlightNode(claimId, {
                    borderColor: '#9c27b0',
                    borderWidth: 2,
                    label: 'CENTRALIZED',
                    labelColor: '#9c27b0'
                });
            });
        }
        
        // Add temporal labels showing publication dates
        if (results.signals?.temporal_clustering?.claim_dates) {
            Object.entries(results.signals.temporal_clustering.claim_dates).forEach(([claimId, date]) => {
                this.addNodeLabel(claimId, new Date(date).toLocaleDateString());
            });
        }
    }
    
    /**
     * Apply anomaly detection highlights
     */
    applyAnomalyHighlights(results) {
        const anomalyColor = '#9c27b0';
        
        // Highlight pattern matches across cultures
        if (results.pattern_matches) {
            results.pattern_matches.forEach(match => {
                this.highlightNode(match.claim_id, {
                    borderColor: anomalyColor,
                    borderWidth: 2,
                    glow: anomalyColor,
                    label: match.culture,
                    labelColor: anomalyColor
                });
            });
        }
        
        // Draw connections between isolated patterns
        if (results.isolated_connections) {
            results.isolated_connections.forEach(connection => {
                this.highlightEdge(connection.from, connection.to, {
                    color: anomalyColor,
                    width: 3,
                    style: 'dashed',
                    label: `${connection.distance_km} km`
                });
            });
        }
        
        // Highlight cultural markers
        if (results.cultural_markers) {
            results.cultural_markers.forEach(marker => {
                this.addNodeLabel(marker.claim_id, `📍 ${marker.culture}`);
            });
        }
        
        // Add geographic distance labels
        if (results.signals?.geographic_isolation?.distances) {
            Object.entries(results.signals.geographic_isolation.distances).forEach(([key, distance]) => {
                const [from, to] = key.split('-');
                this.highlightEdge(from, to, {
                    label: `${distance.toFixed(0)} km`,
                    labelColor: anomalyColor
                });
            });
        }
    }
    
    /**
     * Highlight a specific node
     */
    highlightNode(nodeId, style) {
        if (!this.graphRenderer?.nodes) return;
        
        const node = this.graphRenderer.nodes.find(n => n.id === nodeId);
        if (!node) return;
        
        // Store original style
        if (!this.originalStyles.has(nodeId)) {
            this.originalStyles.set(nodeId, {
                borderColor: node.borderColor,
                borderWidth: node.borderWidth,
                borderStyle: node.borderStyle,
                opacity: node.opacity,
                saturation: node.saturation,
                glow: node.glow,
                brightness: node.brightness
            });
        }
        
        // Apply new style
        Object.assign(node, style);
        
        // Add detection flag
        node._detectionHighlight = true;
    }
    
    /**
     * Highlight an edge between nodes
     */
    highlightEdge(fromId, toId, style) {
        if (!this.graphRenderer?.edges) return;
        
        const edge = this.graphRenderer.edges.find(e => 
            (e.from === fromId && e.to === toId) ||
            (e.from === toId && e.to === fromId)
        );
        
        if (!edge) return;
        
        const edgeKey = `${fromId}-${toId}`;
        
        // Store original style
        if (!this.originalStyles.has(edgeKey)) {
            this.originalStyles.set(edgeKey, {
                color: edge.color,
                width: edge.width,
                style: edge.style
            });
        }
        
        // Apply new style
        Object.assign(edge, style);
        edge._detectionHighlight = true;
    }
    
    /**
     * Highlight connections between cartel members
     */
    highlightCartelConnections(memberIds) {
        if (!this.graphRenderer?.edges) return;
        
        const cartelSet = new Set(memberIds);
        
        this.graphRenderer.edges.forEach(edge => {
            if (cartelSet.has(edge.from) && cartelSet.has(edge.to)) {
                this.highlightEdge(edge.from, edge.to, {
                    color: '#ff5722',
                    width: 4,
                    style: 'solid'
                });
            }
        });
    }
    
    /**
     * Add label to node
     */
    addNodeLabel(nodeId, labelText) {
        if (!this.graphRenderer?.nodes) return;
        
        const node = this.graphRenderer.nodes.find(n => n.id === nodeId);
        if (!node) return;
        
        // Store original label
        if (!this.originalStyles.has(`${nodeId}-label`)) {
            this.originalStyles.set(`${nodeId}-label`, node.label);
        }
        
        // Add detection label
        node.detectionLabel = labelText;
    }
    
    /**
     * Clear all detection highlights
     */
    clearHighlights() {
        if (!this.graphRenderer) return;
        
        // Restore original styles
        this.originalStyles.forEach((originalStyle, key) => {
            if (key.includes('-')) {
                // Edge or label
                if (key.endsWith('-label')) {
                    const nodeId = key.replace('-label', '');
                    const node = this.graphRenderer.nodes?.find(n => n.id === nodeId);
                    if (node) {
                        node.label = originalStyle;
                        delete node.detectionLabel;
                    }
                } else {
                    // Edge
                    const [fromId, toId] = key.split('-');
                    const edge = this.graphRenderer.edges?.find(e =>
                        (e.from === fromId && e.to === toId) ||
                        (e.from === toId && e.to === fromId)
                    );
                    if (edge) {
                        Object.assign(edge, originalStyle);
                        delete edge._detectionHighlight;
                    }
                }
            } else {
                // Node
                const node = this.graphRenderer.nodes?.find(n => n.id === key);
                if (node) {
                    Object.assign(node, originalStyle);
                    delete node._detectionHighlight;
                }
            }
        });
        
        this.originalStyles.clear();
        this.currentHighlights = null;
        
        // Trigger re-render
        this.graphRenderer.render?.();
    }
    
    /**
     * Toggle highlights on/off
     */
    toggleHighlights(enabled) {
        if (enabled && this.currentHighlights) {
            this.applyHighlights(
                this.currentHighlights.results,
                this.currentHighlights.mode
            );
        } else {
            this.clearHighlights();
        }
    }
    
    /**
     * Check if highlights are active
     */
    hasHighlights() {
        return this.currentHighlights !== null;
    }
    
    /**
     * Get current highlight mode
     */
    getMode() {
        return this.currentHighlights?.mode;
    }
}
