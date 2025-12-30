/**
 * Aegis Insight v2.0 - Graph Enhancements
 * Comprehensive module for controls toolbar, node shapes, PNG export, and timeline
 */

import { CONFIG } from '../config.js';

// ============================================================================
// NODE SHAPE RENDERER
// ============================================================================

export class NodeShapeRenderer {
    constructor() {
        this.shapeGenerators = {
            circle: this.createCircle.bind(this),
            square: this.createSquare.bind(this),
            diamond: this.createDiamond.bind(this),
            hexagon: this.createHexagon.bind(this),
            triangle: this.createTriangle.bind(this),
            'circle-hollow': this.createHollowCircle.bind(this)
        };
    }

    /**
     * Get shape and color for a node based on its type and properties
     */
    getNodeVisuals(node) {
        const type = (node.type || 'claim').toLowerCase();
        const claimType = (node.claim_type || '').toUpperCase();
        
        let shape, color, size;
        
        // Determine shape based on node type
        switch (type) {
            case 'entity':
                shape = 'diamond';
                color = '#00ccff';  // Cyan for entities
                size = CONFIG.SIZES.NODE_RADIUS_SMALL;
                break;
            case 'source':
            case 'document':
                shape = 'square';
                color = CONFIG.COLORS.NODE_SOURCE || '#607D8B';
                size = CONFIG.SIZES.NODE_RADIUS_SMALL;
                break;
            case 'container':
            case 'umbrella':
                shape = 'hexagon';
                color = '#9C27B0';  // Purple for containers
                size = CONFIG.SIZES.NODE_RADIUS_LARGE;
                break;
            case 'claim':
            default:
                shape = 'circle';
                // Color based on claim_type
                color = this.getClaimTypeColor(claimType);
                size = CONFIG.SIZES.NODE_RADIUS_SMALL;
                break;
        }
        
        // Adjust for importance
        if (node.importance || node.isImportant) {
            size = CONFIG.SIZES.NODE_RADIUS_LARGE;
        }
        
        return { shape, color, size };
    }
    
    /**
     * Get color for claim type
     */
    getClaimTypeColor(claimType) {
        const colors = {
            'PRIMARY': '#4CAF50',      // Green
            'SECONDARY': '#2196F3',    // Blue
            'META': '#FF9800',         // Orange
            'CONTEXTUAL': '#607D8B',   // Gray-blue
            'UNKNOWN': '#9E9E9E'       // Gray
        };
        return colors[claimType] || colors['UNKNOWN'];
    }

    /**
     * Render a node with the appropriate shape
     */
    renderNode(selection, node) {
        const { shape, color, size } = this.getNodeVisuals(node);
        const generator = this.shapeGenerators[shape] || this.shapeGenerators.circle;
        
        // Clear existing shape
        selection.selectAll('.node-shape').remove();
        
        // Create new shape
        const shapeEl = generator(selection, size);
        shapeEl.attr('class', 'node-shape')
            .attr('fill', color)
            .attr('stroke', '#ffffff')
            .attr('stroke-width', 1.5);
        
        // Add detection highlighting if applicable
        if (node.detected || node.highlighted) {
            shapeEl.attr('stroke', CONFIG.COLORS.SUPPRESSION_HIGH)
                .attr('stroke-width', 3)
                .attr('stroke-dasharray', '5,5');
        }
        
        return shapeEl;
    }

    // Shape generators
    createCircle(selection, size) {
        return selection.append('circle')
            .attr('r', size);
    }

    createSquare(selection, size) {
        const s = size * 1.6;
        return selection.append('rect')
            .attr('width', s)
            .attr('height', s)
            .attr('x', -s / 2)
            .attr('y', -s / 2)
            .attr('rx', 2);
    }

    createDiamond(selection, size) {
        const s = size * 1.4;
        const points = [
            [0, -s],
            [s, 0],
            [0, s],
            [-s, 0]
        ].map(p => p.join(',')).join(' ');
        
        return selection.append('polygon')
            .attr('points', points);
    }

    createHexagon(selection, size) {
        const s = size * 1.2;
        const points = [];
        for (let i = 0; i < 6; i++) {
            const angle = (i * 60 - 30) * Math.PI / 180;
            points.push([
                s * Math.cos(angle),
                s * Math.sin(angle)
            ]);
        }
        return selection.append('polygon')
            .attr('points', points.map(p => p.join(',')).join(' '));
    }

    createTriangle(selection, size) {
        const s = size * 1.5;
        const h = s * Math.sqrt(3) / 2;
        const points = [
            [0, -h * 0.67],
            [s / 2, h * 0.33],
            [-s / 2, h * 0.33]
        ].map(p => p.join(',')).join(' ');
        
        return selection.append('polygon')
            .attr('points', points);
    }

    createHollowCircle(selection, size) {
        return selection.append('circle')
            .attr('r', size)
            .attr('fill', 'none')
            .attr('stroke-width', 3);
    }
}


// ============================================================================
// GRAPH CONTROLS TOOLBAR
// ============================================================================

export class GraphControlsToolbar {
    constructor(graphRenderer, options = {}) {
        this.graph = graphRenderer;
        this.container = options.container || document.getElementById('graph-container');
        this.exporter = new GraphExporter(graphRenderer);
        this.timelineLayout = options.timelineLayout || null;
        this.currentView = 'force';
        
        this.toolbar = null;
    }

    /**
     * Initialize and render the toolbar
     */
    initialize() {
        this.render();
        this.attachEventListeners();
    }

    /**
     * Render toolbar HTML
     */
    render() {
        // Remove existing toolbar
        const existing = this.container.querySelector('.graph-controls-toolbar');
        if (existing) existing.remove();
        
        this.toolbar = document.createElement('div');
        this.toolbar.className = 'graph-controls-toolbar';
        this.toolbar.innerHTML = `
            <div class="toolbar-group view-toggle">
                <button class="toolbar-btn view-btn active" data-view="force" title="Force Layout (F)">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <circle cx="12" cy="12" r="3"/>
                        <circle cx="5" cy="5" r="2"/>
                        <circle cx="19" cy="5" r="2"/>
                        <circle cx="5" cy="19" r="2"/>
                        <circle cx="19" cy="19" r="2"/>
                        <line x1="12" y1="12" x2="5" y2="5" stroke="currentColor" stroke-width="1"/>
                        <line x1="12" y1="12" x2="19" y2="5" stroke="currentColor" stroke-width="1"/>
                        <line x1="12" y1="12" x2="5" y2="19" stroke="currentColor" stroke-width="1"/>
                        <line x1="12" y1="12" x2="19" y2="19" stroke="currentColor" stroke-width="1"/>
                    </svg>
                </button>
                <button class="toolbar-btn view-btn" data-view="timeline" title="Timeline View (T)">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="3" y1="12" x2="21" y2="12"/>
                        <circle cx="6" cy="12" r="2" fill="currentColor"/>
                        <circle cx="12" cy="12" r="2" fill="currentColor"/>
                        <circle cx="18" cy="12" r="2" fill="currentColor"/>
                        <line x1="6" y1="8" x2="6" y2="12"/>
                        <line x1="12" y1="6" x2="12" y2="12"/>
                        <line x1="18" y1="8" x2="18" y2="12"/>
                    </svg>
                </button>
            </div>
            
            <div class="toolbar-separator"></div>
            
            <div class="toolbar-group zoom-controls">
                <button class="toolbar-btn" id="zoom-in-btn" title="Zoom In (+)">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="11" cy="11" r="8"/>
                        <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                        <line x1="11" y1="8" x2="11" y2="14"/>
                        <line x1="8" y1="11" x2="14" y2="11"/>
                    </svg>
                </button>
                <button class="toolbar-btn" id="zoom-out-btn" title="Zoom Out (-)">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="11" cy="11" r="8"/>
                        <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                        <line x1="8" y1="11" x2="14" y2="11"/>
                    </svg>
                </button>
                <button class="toolbar-btn" id="reset-view-btn" title="Reset View (R)">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
                        <path d="M3 3v5h5"/>
                    </svg>
                </button>
                <button class="toolbar-btn" id="fit-btn" title="Fit to View">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>
                    </svg>
                </button>
            </div>
            
            <div class="toolbar-separator"></div>
            
            <div class="toolbar-group action-controls">
                <button class="toolbar-btn" id="export-btn" title="Export as PNG (E)">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="7,10 12,15 17,10"/>
                        <line x1="12" y1="15" x2="12" y2="3"/>
                    </svg>
                </button>
                <button class="toolbar-btn" id="clear-btn" title="Clear Highlights (C)">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="15" y1="9" x2="9" y2="15"/>
                        <line x1="9" y1="9" x2="15" y2="15"/>
                    </svg>
                </button>
            </div>
        `;
        
        this.container.appendChild(this.toolbar);
    }

    /**
     * Attach event listeners
     */
    attachEventListeners() {
        // View toggle buttons
        this.toolbar.querySelectorAll('.view-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.setView(btn.dataset.view);
                this.toolbar.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });
        
        // Zoom controls
        this.toolbar.querySelector('#zoom-in-btn').addEventListener('click', () => {
            this.graph.svg.transition().duration(300)
                .call(this.graph.zoom.scaleBy, 1.3);
        });
        
        this.toolbar.querySelector('#zoom-out-btn').addEventListener('click', () => {
            this.graph.svg.transition().duration(300)
                .call(this.graph.zoom.scaleBy, 0.7);
        });
        
        this.toolbar.querySelector('#reset-view-btn').addEventListener('click', () => {
            if (this.graph.resetView) {
                this.graph.resetView();
            } else {
                this.graph.svg.transition().duration(500)
                    .call(this.graph.zoom.transform, d3.zoomIdentity);
            }
        });
        
        this.toolbar.querySelector('#fit-btn').addEventListener('click', () => {
            this.zoomToFit();
        });
        
        // Action buttons
        this.toolbar.querySelector('#export-btn').addEventListener('click', () => {
            this.exporter.exportPNG('web');
        });
        
        this.toolbar.querySelector('#clear-btn').addEventListener('click', () => {
            if (this.graph.detectionHighlighter) {
                this.graph.detectionHighlighter.clearHighlights();
            }
            if (this.graph.clearHighlight) {
                this.graph.clearHighlight();
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            
            switch (e.key.toLowerCase()) {
                case 'f':
                    this.setView('force');
                    break;
                case 't':
                    this.setView('timeline');
                    break;
                case '+':
                case '=':
                    this.graph.svg.transition().duration(300)
                        .call(this.graph.zoom.scaleBy, 1.3);
                    break;
                case '-':
                    this.graph.svg.transition().duration(300)
                        .call(this.graph.zoom.scaleBy, 0.7);
                    break;
                case 'r':
                    if (this.graph.resetView) this.graph.resetView();
                    break;
                case 'e':
                    this.exporter.exportPNG(e.shiftKey ? 'print' : 'web');
                    break;
                case 'c':
                    if (this.graph.detectionHighlighter) {
                        this.graph.detectionHighlighter.clearHighlights();
                    }
                    break;
            }
        });
    }

    /**
     * Set view mode
     */
    setView(view) {
        if (view === this.currentView) return;
        
        if (view === 'timeline' && this.timelineLayout) {
            this.timelineLayout.activate({
                groupBy: 'source',
                scaleType: 'auto'
            });
        } else if (this.timelineLayout) {
            this.timelineLayout.deactivate();
        }
        
        this.currentView = view;
        
        // Update button states
        this.toolbar.querySelectorAll('.view-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === view);
        });
    }

    /**
     * Zoom to fit all nodes
     */
    zoomToFit() {
        const svg = this.graph.svg;
        const g = this.graph.g;
        const bounds = g.node().getBBox();
        const parent = svg.node().parentElement;
        const fullWidth = parent.clientWidth;
        const fullHeight = parent.clientHeight;
        
        const width = bounds.width;
        const height = bounds.height;
        const midX = bounds.x + width / 2;
        const midY = bounds.y + height / 2;
        
        if (width === 0 || height === 0) return;
        
        const scale = 0.85 / Math.max(width / fullWidth, height / fullHeight);
        const translate = [fullWidth / 2 - scale * midX, fullHeight / 2 - scale * midY];
        
        svg.transition().duration(750)
            .call(this.graph.zoom.transform, 
                d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale));
    }
}


// ============================================================================
// GRAPH EXPORTER (PNG)
// ============================================================================

export class GraphExporter {
    constructor(graphRenderer) {
        this.graph = graphRenderer;
    }

    /**
     * Export graph as PNG
     * @param {string} theme - 'web' (dark) or 'print' (light)
     */
    async exportPNG(theme = 'web') {
        const svg = this.graph.svg.node();
        const themeConfig = CONFIG.EXPORT?.THEMES?.[theme.toUpperCase()] || {
            background: theme === 'print' ? '#ffffff' : '#0a0e17',
            text: theme === 'print' ? '#000000' : '#e0e0e0'
        };
        
        try {
            // Get SVG dimensions
            const bbox = svg.getBoundingClientRect();
            const width = bbox.width || 800;
            const height = bbox.height || 600;
            
            // Create canvas with 2x resolution for crisp export
            const canvas = document.createElement('canvas');
            const scale = 2;
            canvas.width = width * scale;
            canvas.height = height * scale;
            const ctx = canvas.getContext('2d');
            ctx.scale(scale, scale);
            
            // Draw background
            ctx.fillStyle = themeConfig.background;
            ctx.fillRect(0, 0, width, height);
            
            // Clone and prepare SVG
            const svgClone = svg.cloneNode(true);
            
            // Apply theme adjustments
            if (theme === 'print') {
                this.applyPrintTheme(svgClone);
            }
            
            // Convert SVG to data URL
            const svgData = new XMLSerializer().serializeToString(svgClone);
            const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
            const svgUrl = URL.createObjectURL(svgBlob);
            
            // Load and draw image
            const img = new Image();
            
            await new Promise((resolve, reject) => {
                img.onload = () => {
                    ctx.drawImage(img, 0, 0, width, height);
                    URL.revokeObjectURL(svgUrl);
                    resolve();
                };
                img.onerror = reject;
                img.src = svgUrl;
            });
            
            // Add metadata watermark
            this.addMetadata(ctx, width, height, themeConfig);
            
            // Download
            const filename = this.generateFilename();
            canvas.toBlob((blob) => {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                a.click();
                URL.revokeObjectURL(url);
            }, 'image/png', 1.0);
            
            console.log(`Graph exported as ${filename}`);
            
        } catch (error) {
            console.error('Export failed:', error);
            alert('Failed to export graph. Please try again.');
        }
    }

    /**
     * Apply print-friendly theme
     */
    applyPrintTheme(svg) {
        // Change background
        svg.style.backgroundColor = '#ffffff';
        
        // Change text color
        svg.querySelectorAll('text').forEach(text => {
            text.style.fill = '#000000';
        });
        
        // Darken edges
        svg.querySelectorAll('line').forEach(line => {
            const currentStroke = line.getAttribute('stroke');
            if (currentStroke && currentStroke !== 'none') {
                // Darken the color
                line.setAttribute('stroke', this.darkenColor(currentStroke));
            }
        });
    }

    /**
     * Darken a color for print
     */
    darkenColor(color) {
        // Simple darkening - convert light colors to darker versions
        if (color.startsWith('#')) {
            const r = parseInt(color.slice(1, 3), 16);
            const g = parseInt(color.slice(3, 5), 16);
            const b = parseInt(color.slice(5, 7), 16);
            
            // If color is light, darken it
            const brightness = (r + g + b) / 3;
            if (brightness > 128) {
                return `rgb(${Math.floor(r * 0.5)}, ${Math.floor(g * 0.5)}, ${Math.floor(b * 0.5)})`;
            }
        }
        return color;
    }

    /**
     * Add metadata to exported image
     */
    addMetadata(ctx, width, height, theme) {
        ctx.font = '10px system-ui, sans-serif';
        ctx.fillStyle = theme.text === '#000000' ? 'rgba(0,0,0,0.5)' : 'rgba(255,255,255,0.5)';
        ctx.textAlign = 'right';
        
        const timestamp = new Date().toLocaleString();
        const nodeCount = this.graph.data?.nodes?.length || 0;
        const edgeCount = this.graph.data?.links?.length || 0;
        
        ctx.fillText(`Aegis Insight | ${nodeCount} nodes, ${edgeCount} edges | ${timestamp}`, width - 10, height - 10);
    }

    /**
     * Generate filename for export
     */
    generateFilename() {
        const prefix = CONFIG.EXPORT?.FILENAME_PREFIX || 'aegis_graph';
        const date = new Date().toISOString().split('T')[0];
        const query = this.graph.lastQuery || 'graph';
        const safeQuery = query.replace(/[^a-z0-9]/gi, '_').substring(0, 30);
        
        return `${prefix}_${safeQuery}_${date}.png`;
    }
}


// ============================================================================
// NODE LEGEND
// ============================================================================

export class NodeLegend {
    constructor(container) {
        this.container = container;
    }

    /**
     * Render legend showing node shape/color meanings
     */
    render() {
        const legendEl = document.createElement('div');
        legendEl.className = 'node-legend';
        legendEl.innerHTML = `
            <div class="legend-title">Node Types</div>
            <div class="legend-items">
                <div class="legend-item">
                    <svg width="20" height="20"><circle cx="10" cy="10" r="6" fill="#4CAF50" stroke="#fff" stroke-width="1"/></svg>
                    <span>PRIMARY Claim</span>
                </div>
                <div class="legend-item">
                    <svg width="20" height="20"><circle cx="10" cy="10" r="6" fill="#2196F3" stroke="#fff" stroke-width="1"/></svg>
                    <span>SECONDARY Claim</span>
                </div>
                <div class="legend-item">
                    <svg width="20" height="20"><circle cx="10" cy="10" r="6" fill="#FF9800" stroke="#fff" stroke-width="1"/></svg>
                    <span>META Claim</span>
                </div>
                <div class="legend-item">
                    <svg width="20" height="20"><polygon points="10,3 17,10 10,17 3,10" fill="#00ccff" stroke="#fff" stroke-width="1"/></svg>
                    <span>Entity</span>
                </div>
                <div class="legend-item">
                    <svg width="20" height="20"><rect x="4" y="4" width="12" height="12" rx="2" fill="#607D8B" stroke="#fff" stroke-width="1"/></svg>
                    <span>Source/Document</span>
                </div>
            </div>
        `;
        
        // Remove existing legend
        const existing = this.container.querySelector('.node-legend');
        if (existing) existing.remove();
        
        this.container.appendChild(legendEl);
    }
}


// ============================================================================
// INTEGRATION HELPER
// ============================================================================

/**
 * Initialize all graph enhancements
 * Call this after GraphRenderer is initialized
 */
export function initializeGraphEnhancements(graphRenderer, options = {}) {
    const container = options.container || document.getElementById('graph-container');
    
    // Initialize node shape renderer
    const shapeRenderer = new NodeShapeRenderer();
    graphRenderer.shapeRenderer = shapeRenderer;
    
    // Initialize controls toolbar
    const toolbar = new GraphControlsToolbar(graphRenderer, {
        container: container,
        timelineLayout: options.timelineLayout || null
    });
    toolbar.initialize();
    graphRenderer.controlsToolbar = toolbar;
    
    // Initialize legend
    const legend = new NodeLegend(container);
    legend.render();
    graphRenderer.legend = legend;
    
    // Override node rendering if needed
    if (options.useShapes !== false) {
        patchNodeRendering(graphRenderer, shapeRenderer);
    }
    
    console.log('✓ Graph enhancements initialized');
    
    return {
        shapeRenderer,
        toolbar,
        legend
    };
}

/**
 * Patch GraphRenderer to use custom shapes
 */
function patchNodeRendering(graphRenderer, shapeRenderer) {
    // Store original render method
    const originalRender = graphRenderer.render.bind(graphRenderer);
    
    // Override render to apply shapes after standard render
    graphRenderer.render = function(data) {
        // Call original render
        originalRender(data);
        
        // Apply shapes to all nodes
        if (this.nodesGroup) {
            this.nodesGroup.selectAll('g.node').each(function(d) {
                const node = d3.select(this);
                shapeRenderer.renderNode(node, d);
            });
        }
    };
}
