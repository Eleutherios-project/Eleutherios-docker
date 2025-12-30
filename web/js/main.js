/**
 * Aegis Insight v2.0 - Main Integration
 * Coordinates graph renderer, detection highlighting, and timeline layout
 */

import { CONFIG } from './config.js';
import { GraphRenderer } from './graph/GraphRenderer.js';
import { DetectionHighlighter } from './graph/DetectionHighlighter.js';
import { TimelineLayout } from './graph/TimelineLayout.js';
import { DetectionControls } from './detection/DetectionControls.js';
import { GraphFilter } from './graph/GraphFilter.js';

/**
 * Main Aegis application class
 */
export class AegisApp {
    constructor(options = {}) {
        this.options = {
            graphContainer: options.graphContainer || '#graph-container',
            controlsContainer: options.controlsContainer || '#detection-panel',
            apiBaseUrl: options.apiBaseUrl || 'http://localhost:8001',
            ...options
        };
        
        // Core components
        this.graphRenderer = null;
        this.detectionHighlighter = null;
        this.timelineLayout = null;
        this.detectionControls = null;
        this.graphFilter = null;
        
        // State
        this.currentView = 'force';  // 'force' or 'timeline'
        this.currentMode = 'standard';
    }

    /**
     * Initialize the application
     */
    async initialize() {
        console.log('Initializing Aegis Insight v2.0...');
        
        // Initialize graph renderer
        this.graphRenderer = new GraphRenderer(this.options.graphContainer);
        
        // Initialize detection highlighter
        this.detectionHighlighter = new DetectionHighlighter(this.graphRenderer);
        
        // Initialize timeline layout
        this.timelineLayout = new TimelineLayout(this.graphRenderer);
        
        // Initialize graph filter
        this.graphFilter = new GraphFilter(this.graphRenderer, {
            container: document.querySelector(this.options.graphContainer),
            onFilterChange: (nodes, edges) => this.handleFilterChange(nodes, edges)
        });
        this.graphFilter.createFilterPanel();
        
        // Initialize detection controls
        this.detectionControls = new DetectionControls({
            containerId: this.options.controlsContainer,
            apiBaseUrl: this.options.apiBaseUrl,
            onDetectionComplete: (results, mode) => this.handleDetectionComplete(results, mode),
            onModeChange: (mode) => this.handleModeChange(mode)
        });
        this.detectionControls.initialize();
        
        // Add graph controls toolbar
        this.addGraphControls();
        
        // Add keyboard shortcuts
        this.setupKeyboardShortcuts();
        
        console.log('Aegis Insight initialized');
    }

    /**
     * Load and render graph data
     * @param {Object} data - { nodes: [], links: [] }
     */
    loadGraph(data) {
        this.graphRenderer.render(data);
        
        // Set up node click handler
        this.graphRenderer.onNodeClick((node) => {
            this.handleNodeClick(node);
        });
    }

    /**
     * Handle detection complete
     */
    handleDetectionComplete(results, mode) {
        console.log('Detection complete:', mode, results);
        
        if (mode !== 'standard' && results) {
            // Pass detection results to graph filter for clustering
            if (this.graphFilter) {
                this.graphFilter.setDetectionResults(results);
            }
            
            // Apply visual highlighting
            this.detectionHighlighter.applyHighlights(results, mode);
            
            // If we have graph data in results, update the graph
            if (results.graph_data) {
                this.graphRenderer.render(results.graph_data);
            }
        }
    }

    /**
     * Handle mode change
     */
    handleModeChange(mode) {
        this.currentMode = mode;
        
        // Clear existing highlights when mode changes
        if (mode === 'standard') {
            this.detectionHighlighter.clearHighlights();
        }
    }

    /**
     * Handle filter changes from GraphFilter
     */
    handleFilterChange(nodes, edges) {
        console.log(`Filter applied: ${nodes.length} nodes, ${edges.length} edges`);
        
        // Update graph with filtered data
        if (this.graphRenderer) {
            this.graphRenderer.render({ nodes, edges });
            
            // Re-apply detection highlights after filter
            if (this.currentMode !== 'standard' && this.detectionControls?.lastResults) {
                const results = this.detectionControls.lastResults.result || this.detectionControls.lastResults;
                this.detectionHighlighter.applyHighlights(results, this.currentMode);
            }
            
            // Update cluster boundary if enabled
            if (this.graphFilter?.clusteringEnabled) {
                this.graphFilter.renderClusterBoundary(this.graphRenderer.svg);
            }
        }
    }
    
    /**
     * Handle node click
     */
    handleNodeClick(node) {
        console.log('Node clicked:', node);
        
        // Highlight connections
        this.graphRenderer.highlightConnections(node.id);
        
        // Emit event for external handlers
        const event = new CustomEvent('aegis:nodeClick', { detail: node });
        document.dispatchEvent(event);
    }

    /**
     * Add graph controls toolbar
     */
    addGraphControls() {
        const container = document.querySelector(this.options.graphContainer);
        if (!container) return;
        
        // Create controls HTML
        const controls = document.createElement('div');
        controls.className = 'graph-controls';
        controls.innerHTML = `
            <div class="view-toggle">
                <button class="view-btn active" data-view="force" title="Force Layout">
                    🕸️ Graph
                </button>
                <button class="view-btn" data-view="timeline" title="Timeline View">
                    📅 Timeline
                </button>
            </div>
            <button class="control-btn" id="zoom-in-btn" title="Zoom In">🔍+</button>
            <button class="control-btn" id="zoom-out-btn" title="Zoom Out">🔍−</button>
            <button class="control-btn" id="reset-view-btn" title="Reset View">⟲</button>
            <button class="control-btn" id="fit-btn" title="Fit to View">⊙</button>
            <button class="control-btn" id="export-btn" title="Export PNG">💾</button>
            <button class="control-btn" id="clear-btn" title="Clear Highlights">✕</button>
        `;
        
        container.style.position = 'relative';
        container.appendChild(controls);
        
        // Attach handlers
        this.attachControlHandlers(controls);
    }

    /**
     * Attach control button handlers
     */
    attachControlHandlers(controls) {
        // View toggle
        controls.querySelectorAll('.view-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.setView(btn.dataset.view);
                controls.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });
        
        // Zoom controls
        controls.querySelector('#zoom-in-btn').addEventListener('click', () => {
            this.graphRenderer.svg.transition().duration(300)
                .call(this.graphRenderer.zoom.scaleBy, 1.3);
        });
        
        controls.querySelector('#zoom-out-btn').addEventListener('click', () => {
            this.graphRenderer.svg.transition().duration(300)
                .call(this.graphRenderer.zoom.scaleBy, 0.7);
        });
        
        controls.querySelector('#reset-view-btn').addEventListener('click', () => {
            this.graphRenderer.resetView();
        });
        
        controls.querySelector('#fit-btn').addEventListener('click', () => {
            this.graphRenderer.zoomToFit();
        });
        
        controls.querySelector('#export-btn').addEventListener('click', () => {
            this.exportGraphPNG();
        });
        
        controls.querySelector('#clear-btn').addEventListener('click', () => {
            this.detectionHighlighter.clearHighlights();
            this.graphRenderer.clearHighlight();
        });
    }

    /**
     * Set view mode (force or timeline)
     */
    setView(view) {
        if (view === this.currentView) return;
        
        if (view === 'timeline') {
            this.timelineLayout.activate({
                groupBy: 'source',
                scaleType: 'auto'
            });
        } else {
            this.timelineLayout.deactivate();
        }
        
        this.currentView = view;
    }

    /**
     * Toggle timeline view
     */
    toggleTimeline() {
        this.setView(this.currentView === 'timeline' ? 'force' : 'timeline');
    }

    /**
     * Export graph as PNG
     */
    exportGraphPNG() {
        const svg = this.graphRenderer.svg.node();
        
        // Create canvas
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        // Get SVG dimensions
        const bbox = svg.getBoundingClientRect();
        canvas.width = bbox.width * 2;  // 2x for retina
        canvas.height = bbox.height * 2;
        ctx.scale(2, 2);
        
        // Draw background
        ctx.fillStyle = CONFIG.COLORS.BACKGROUND;
        ctx.fillRect(0, 0, bbox.width, bbox.height);
        
        // Convert SVG to image
        const svgData = new XMLSerializer().serializeToString(svg);
        const img = new Image();
        
        img.onload = () => {
            ctx.drawImage(img, 0, 0, bbox.width, bbox.height);
            
            // Download
            const a = document.createElement('a');
            a.download = `aegis-graph-${Date.now()}.png`;
            a.href = canvas.toDataURL('image/png');
            a.click();
        };
        
        img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgData)));
    }

    /**
     * Setup keyboard shortcuts
     */
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ignore if typing in input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            
            switch (e.key) {
                case 't':
                case 'T':
                    this.toggleTimeline();
                    break;
                case 'Escape':
                    this.detectionHighlighter.clearHighlights();
                    this.graphRenderer.clearHighlight();
                    break;
                case 'f':
                case 'F':
                    this.graphRenderer.zoomToFit();
                    break;
                case 'r':
                case 'R':
                    this.graphRenderer.resetView();
                    break;
            }
        });
    }

    /**
     * Search the graph
     */
    async search(query) {
        // Delegate to detection controls
        this.detectionControls.searchInput.value = query;
        await this.detectionControls.executeSearch();
    }

    /**
     * Run detection
     */
    async runDetection(topic, mode = 'suppression') {
        this.detectionControls.setMode(mode);
        this.detectionControls.searchInput.value = topic;
        await this.detectionControls.executeSearch();
    }

    /**
     * Get graph statistics
     */
    getStats() {
        return this.graphRenderer.getStats();
    }

    /**
     * Destroy application
     */
    destroy() {
        this.timelineLayout.deactivate();
        this.graphRenderer.destroy();
        this.detectionHighlighter.clearHighlights();
    }
}


/**
 * Global initialization
 * Call this when DOM is ready
 */
export async function initializeAegis(options = {}) {
    const app = new AegisApp(options);
    await app.initialize();
    
    // Make available globally for debugging
    window.aegis = app;
    
    return app;
}


// Auto-initialize if running as module with default options
if (typeof window !== 'undefined') {
    window.AegisApp = AegisApp;
    window.initializeAegis = initializeAegis;
}
