/**
 * Aegis Insight v2.0 - Graph Renderer
 * Main visualization engine that coordinates all graph components
 */

import { CONFIG } from '../config.js';
import { ColorScheme } from './ColorScheme.js';
import { NodeShapes } from './NodeShapes.js';
import { ForceSimulation } from './ForceSimulation.js';

export class GraphRenderer {
    constructor(containerId) {
        this.container = document.querySelector(containerId);
        if (!this.container) {
            throw new Error(`Container ${containerId} not found`);
        }
        
        // Get dimensions with fallback (container might be hidden initially)
        this.width = this.container.clientWidth || 800;
        this.height = this.container.clientHeight || 600;

        // Ensure minimum dimensions
        if (this.width < 100) this.width = 800;
        if (this.height < 100) this.height = 600;

        console.log('GraphRenderer dimensions:', this.width, 'x', this.height);
        
        this.svg = null;
        this.g = null; // Main group for zoom/pan
        this.linksGroup = null;
        this.nodesGroup = null;
        
        this.simulation = new ForceSimulation(this.width, this.height);
        this.zoom = null;
        
        this.data = { nodes: [], links: [] };
        this.selectedNode = null;
        this.onNodeClickCallback = null;
        
        this.initializeSVG();
    }
    
    /**
     * Initialize SVG canvas
     */
    initializeSVG() {
        // Clear existing
        this.container.innerHTML = '';
        
        // Create SVG
        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('viewBox', `0 0 ${this.width} ${this.height}`);
        
        // Add defs for filters (suppression glow)
        const defs = this.svg.append('defs');
        const filter = defs.append('filter')
            .attr('id', 'suppression-glow')
            .attr('x', '-50%')
            .attr('y', '-50%')
            .attr('width', '200%')
            .attr('height', '200%');
        
        filter.append('feGaussianBlur')
            .attr('stdDeviation', CONFIG.SUPPRESSION.GLOW_RADIUS)
            .attr('result', 'coloredBlur');
        
        const feMerge = filter.append('feMerge');
        feMerge.append('feMergeNode').attr('in', 'coloredBlur');
        feMerge.append('feMergeNode').attr('in', 'SourceGraphic');
        
        // Create main group for zoom/pan
        this.g = this.svg.append('g')
            .attr('class', 'main-group');
        
        // Create groups for links and nodes (links behind nodes)
        this.linksGroup = this.g.append('g')
            .attr('class', 'links');
        
        this.nodesGroup = this.g.append('g')
            .attr('class', 'nodes');
        
        // Set up zoom behavior
        this.setupZoom();
    }
    
    /**
     * Set up zoom and pan behavior
     */
    setupZoom() {
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                this.g.attr('transform', event.transform);
            });
        
        this.svg.call(this.zoom);
    }
    
    /**
     * Create drag behavior for nodes
     */
    createDragBehavior() {
        return d3.drag()
            .on('start', (event, d) => {
                if (!event.active) this.simulation.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) this.simulation.simulation.alphaTarget(0);
                // Don't clear fx/fy - let node stay where user dragged it
                // d.fx = null;
                // d.fy = null;
            });
    }

    /**
     * Create tooltip text for a node
     * @param {Object} d - Node data
     * @returns {string} Formatted tooltip text
     */
    getTooltipText(d) {
        const parts = [];

        // Type and ID
        parts.push(`Type: ${d.type || 'Unknown'}`);

        // Main content
        if (d.claim_text) {
            // For claims - show full text
            parts.push(`\nClaim: ${d.claim_text}`);
        } else if (d.name) {
            parts.push(`\nName: ${d.name}`);
        }

        // Confidence/Trust
        if (d.confidence !== undefined) {
            parts.push(`\nConfidence: ${(d.confidence * 100).toFixed(0)}%`);
        } else if (d.trust_score !== undefined) {
            parts.push(`\nTrust: ${(d.trust_score * 100).toFixed(0)}%`);
        }

        // Claim type
        if (d.claim_type) {
            parts.push(`\nClaim Type: ${d.claim_type}`);
        }

        // Suppression
        if (d.suppression_score && d.suppression_score > 0.3) {
            parts.push(`\n⚠️ Suppression: ${(d.suppression_score * 100).toFixed(0)}%`);
        }

        // Source file (truncated)
        if (d.source_file) {
            const filename = d.source_file.split('/').pop();
            parts.push(`\nSource: ${filename}`);
        }

        return parts.join('');
    }

    /**
     * Render graph with given data
     * @param {Object} data - { nodes: [], links: [] }
     */
    render(data) {
        this.data = data;

        // Prepare data
        this.prepareData();

        // Initialize node positions (prevent NaN errors)
        this.data.nodes.forEach((node, i) => {
            node.x = node.x || this.width / 2 + Math.random() * 100 - 50;
            node.y = node.y || this.height / 2 + Math.random() * 100 - 50;
        });

        // Render edges
        this.renderEdges();

        // Render nodes
        this.renderNodes();

        // Initialize/update simulation
        this.simulation.update(this.data.nodes, this.data.links);
        this.simulation.onTick(() => this.updatePositions());

        // Adjust forces based on node count
        this.simulation.adjustForNodeCount(this.data.nodes.length);
    }

    /**
     * Prepare data (ensure IDs, add missing properties)
     */
    prepareData() {
        // Fix: API returns 'edges' but we expect 'links'
        if (this.data.edges && !this.data.links) {
            this.data.links = this.data.edges;
        }

        // Ensure arrays exist (prevent forEach errors on undefined)
        if (!this.data.nodes) this.data.nodes = [];
        if (!this.data.links) this.data.links = [];

        // Ensure all nodes have required properties
        this.data.nodes.forEach(node => {
            if (!node.id) node.id = node.name || Math.random().toString(36);
            if (!node.type) node.type = 'entity';
            if (!node.label) node.label = node.name || node.id;
        });

        // Ensure all links have source/target
        this.data.links.forEach(link => {
            if (typeof link.source === 'string') {
                link.source = this.data.nodes.find(n => n.id === link.source);
            }
            if (typeof link.target === 'string') {
                link.target = this.data.nodes.find(n => n.id === link.target);
            }
        });
    }

    /**
     * Render edges
     */
    renderEdges() {
        const link = this.linksGroup
            .selectAll('line')
            .data(this.data.links, d => `${d.source.id}-${d.target.id}`);

        // Exit
        link.exit().remove();

        // Enter + Update
        link.enter()
            .append('line')
            .merge(link)
            .attr('stroke', d => ColorScheme.getEdgeColor(d))
            .attr('stroke-width', d => ColorScheme.getEdgeWidth(d))
            .attr('stroke-dasharray', d => ColorScheme.getEdgeDasharray(d))
            .attr('stroke-opacity', d => ColorScheme.getEdgeOpacity(d));
    }

    /**
     * Render nodes
     */
    renderNodes() {
        const node = this.nodesGroup
            .selectAll('g.node')
            .data(this.data.nodes, d => d.id);

        // Exit
        node.exit().remove();

        // Enter
        const nodeEnter = node.enter()
            .append('g')
            .attr('class', 'node')
            .style('cursor', 'grab')
            .on('click', (event, d) => this.handleNodeClick(event, d))
            .call(this.createDragBehavior());

        // Add tooltip (native SVG title element)
        nodeEnter.append('title')
            .text(d => this.getTooltipText(d));

        // Change cursor when dragging
        nodeEnter.on('mousedown', function() {
            d3.select(this).style('cursor', 'grabbing');
        });
        nodeEnter.on('mouseup', function() {
            d3.select(this).style('cursor', 'grab');
        });

        // Merge and render shapes
        const nodeMerge = nodeEnter.merge(node);

        // Update tooltips for existing nodes too
        nodeMerge.select('title')
            .text(d => this.getTooltipText(d));

        NodeShapes.renderNode(nodeMerge);
    }

    /**
     * Update positions on simulation tick
     */
    updatePositions() {
        // Update link positions
        this.linksGroup.selectAll('line')
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        // Update node positions
        const nodes = this.nodesGroup.selectAll('g.node');
        nodes.attr('transform', d => `translate(${d.x},${d.y})`);

        // Update label visibility based on velocity (LOD)
        NodeShapes.updateLabelVisibility(nodes);
    }

    /**
     * Handle node click
     * @param {Event} event - Click event
     * @param {Object} node - Node data
     */
    handleNodeClick(event, node) {
        event.stopPropagation();

        // Update selection
        this.selectedNode = node;

        // Visual feedback
        this.nodesGroup.selectAll('g.node')
            .classed('selected', d => d.id === node.id);

        // Callback
        if (this.onNodeClickCallback) {
            this.onNodeClickCallback(node);
        }
    }

    /**
     * Register node click callback
     * @param {Function} callback - Function to call on node click
     */
    onNodeClick(callback) {
        this.onNodeClickCallback = callback;
    }

    /**
     * Clear selection
     */
    clearSelection() {
        this.selectedNode = null;
        this.nodesGroup.selectAll('g.node').classed('selected', false);
    }

    /**
     * Reset view (zoom and pan)
     */
    resetView() {
        this.svg.transition()
            .duration(750)
            .call(this.zoom.transform, d3.zoomIdentity);
    }

    /**
     * Zoom to fit all nodes
     */
    zoomToFit(padding = 50) {
        if (!this.data.nodes.length) return;

        const bounds = this.getBounds();
        const width = bounds.maxX - bounds.minX;
        const height = bounds.maxY - bounds.minY;

        const scale = 0.9 / Math.max(width / this.width, height / this.height);
        const translate = [
            this.width / 2 - scale * (bounds.minX + bounds.maxX) / 2,
            this.height / 2 - scale * (bounds.minY + bounds.maxY) / 2
        ];

        this.svg.transition()
            .duration(750)
            .call(this.zoom.transform, d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale));
    }

    /**
     * Get bounds of all nodes
     * @returns {Object} {minX, maxX, minY, maxY}
     */
    getBounds() {
        const xs = this.data.nodes.map(d => d.x);
        const ys = this.data.nodes.map(d => d.y);

        return {
            minX: Math.min(...xs),
            maxX: Math.max(...xs),
            minY: Math.min(...ys),
            maxY: Math.max(...ys)
        };
    }

    /**
     * Highlight nodes connected to given node
     * @param {string} nodeId - Node ID to highlight connections for
     */
    highlightConnections(nodeId) {
        const connectedIds = new Set();
        connectedIds.add(nodeId);

        // Find connected nodes
        this.data.links.forEach(link => {
            if (link.source.id === nodeId) connectedIds.add(link.target.id);
            if (link.target.id === nodeId) connectedIds.add(link.source.id);
        });

        // Highlight connected nodes
        this.nodesGroup.selectAll('g.node')
            .classed('connected', d => connectedIds.has(d.id))
            .style('opacity', d => connectedIds.has(d.id) ? 1 : 0.3);

        // Highlight connected edges
        this.linksGroup.selectAll('line')
            .style('opacity', d => {
                const connected = d.source.id === nodeId || d.target.id === nodeId;
                return connected ? ColorScheme.getEdgeOpacity(d) : 0.1;
            });
    }

    /**
     * Clear highlighting
     */
    clearHighlight() {
        this.nodesGroup.selectAll('g.node')
            .classed('connected', false)
            .style('opacity', 1);

        this.linksGroup.selectAll('line')
            .style('opacity', d => ColorScheme.getEdgeOpacity(d));
    }

    /**
     * Get current graph statistics
     * @returns {Object} Statistics object
     */
    getStats() {
        const trustScores = this.data.nodes
            .map(n => n.trust_score || n.confidence)
            .filter(t => t !== undefined && t !== null);

        const avgTrust = trustScores.length > 0
            ? trustScores.reduce((a, b) => a + b, 0) / trustScores.length
            : 0;

        const suppressedCount = this.data.nodes
            .filter(n => (n.suppression_score || 0) > CONFIG.SUPPRESSION.HIGH_THRESHOLD)
            .length;

        return {
            nodeCount: this.data.nodes.length,
            edgeCount: this.data.links.length,
            avgTrust: avgTrust,
            suppressedCount: suppressedCount,
            isRunning: this.simulation.isRunning()
        };
    }

    /**
     * Destroy renderer and clean up
     */
    destroy() {
        this.simulation.stop();
        if (this.svg) {
            this.svg.remove();
        }
    }
}