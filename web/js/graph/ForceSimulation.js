/**
 * Aegis Insight v2.0 - Force Simulation
 * D3 force simulation with performance optimizations
 */

import { CONFIG } from '../config.js';
import { NodeShapes } from './NodeShapes.js';

export class ForceSimulation {
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.simulation = null;
        this.tickCount = 0;
        this.onTickCallback = null;
    }
    
    /**
     * Initialize force simulation
     * @param {Array} nodes - Array of node objects
     * @param {Array} links - Array of link objects
     * @returns {d3.Simulation} D3 force simulation
     */
    initialize(nodes, links) {
        this.simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links)
                .id(d => d.id)
                .distance(CONFIG.SIMULATION.LINK_DISTANCE))
            .force('charge', d3.forceManyBody()
                .strength(CONFIG.SIMULATION.CHARGE_STRENGTH))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide()
                .radius(d => NodeShapes.getNodeSize(d) + 5))
            .alphaTarget(CONFIG.SIMULATION.ALPHA_TARGET)
            .alphaDecay(CONFIG.SIMULATION.ALPHA_DECAY)
            .velocityDecay(CONFIG.SIMULATION.VELOCITY_DECAY);
        
        // Set up throttled tick handler
        this.simulation.on('tick', () => this.handleTick());
        
        return this.simulation;
    }
    
    /**
     * Handle simulation tick with throttling
     */
    handleTick() {
        this.tickCount++;
        
        // Only update every Nth frame (performance optimization)
        if (this.tickCount % CONFIG.SIMULATION.TICK_THROTTLE === 0) {
            if (this.onTickCallback) {
                this.onTickCallback();
            }
        }
    }
    
    /**
     * Register callback for tick events
     * @param {Function} callback - Function to call on tick
     */
    onTick(callback) {
        this.onTickCallback = callback;
    }
    
    /**
     * Update forces based on new data
     * @param {Array} nodes - Updated nodes
     * @param {Array} links - Updated links
     */
    update(nodes, links) {
        if (!this.simulation) {
            return this.initialize(nodes, links);
        }
        
        this.simulation.nodes(nodes);
        this.simulation.force('link').links(links);
        this.simulation.alpha(1).restart();
    }
    
    /**
     * Add boundary containment force for umbrella containers
     * @param {Array} nodes - All nodes
     */
    addBoundaryForce(nodes) {
        this.simulation.force('boundary', alpha => {
            nodes.forEach(node => {
                if (node.cluster) {
                    // Find the umbrella node this belongs to
                    const umbrella = nodes.find(n => n.id === node.cluster);
                    if (!umbrella || umbrella.type !== 'umbrella') return;
                    
                    // Calculate distance from umbrella center
                    const dx = node.x - umbrella.x;
                    const dy = node.y - umbrella.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    
                    const umbrellaRadius = umbrella.radius || CONFIG.SIZES.NODE_RADIUS_CONTAINER;
                    const nodeRadius = NodeShapes.getNodeSize(node);
                    
                    // If outside boundary, pull back in
                    if (distance > umbrellaRadius - nodeRadius - 10) {
                        const strength = 0.1 * alpha;
                        node.vx -= dx * strength;
                        node.vy -= dy * strength;
                    }
                }
            });
        });
    }
    
    /**
     * Stop simulation
     */
    stop() {
        if (this.simulation) {
            this.simulation.stop();
        }
    }
    
    /**
     * Restart simulation with alpha
     * @param {number} alpha - Alpha value (default 1)
     */
    restart(alpha = 1) {
        if (this.simulation) {
            this.simulation.alpha(alpha).restart();
        }
    }
    
    /**
     * Get simulation alpha (animation progress)
     * @returns {number} Current alpha value
     */
    getAlpha() {
        return this.simulation ? this.simulation.alpha() : 0;
    }
    
    /**
     * Check if simulation is still running
     * @returns {boolean}
     */
    isRunning() {
        return this.getAlpha() > 0.005;
    }
    
    /**
     * Apply drag behavior to nodes
     * @returns {d3.Drag} D3 drag behavior
     */
    createDragBehavior() {
        const simulation = this.simulation;
        
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
        
        return d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended);
    }
    
    /**
     * Adjust forces based on graph size (performance optimization)
     * @param {number} nodeCount - Number of nodes
     */
    adjustForNodeCount(nodeCount) {
        if (!this.simulation) return;
        
        // Reduce charge strength for large graphs
        let chargeStrength = CONFIG.SIMULATION.CHARGE_STRENGTH;
        if (nodeCount > 100) {
            chargeStrength *= 0.5;
        }
        if (nodeCount > 200) {
            chargeStrength *= 0.5;
        }
        
        this.simulation.force('charge').strength(chargeStrength);
    }
    
    /**
     * Reheat simulation (useful after adding/removing nodes)
     * @param {number} alpha - Target alpha (default 0.3)
     */
    reheat(alpha = 0.3) {
        if (this.simulation) {
            this.simulation.alpha(alpha).restart();
        }
    }
}
