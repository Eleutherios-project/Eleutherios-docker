/**
 * Aegis Insight v2.0 - Color Scheme
 * Determines node colors based on type, trust score, and position
 */

import { CONFIG, getTrustColor, getPositionColor } from '../config.js';

export class ColorScheme {
    /**
     * Get the appropriate color for a node based on its properties
     * @param {Object} node - Node data object
     * @returns {string} Hex color code
     */
    static getNodeColor(node) {
        // Claims use position-based colors (if position is specified)
        if (node.type === 'claim' && node.position) {
            return getPositionColor(node.position);
        }
        
        // Source documents get fixed color
        if (node.type === 'source' || node.type === 'document') {
            return CONFIG.COLORS.NODE_SOURCE;
        }
        
        // Umbrella containers can have custom colors
        if (node.type === 'umbrella') {
            return node.color || CONFIG.COLORS.ACCENT;
        }
        
        // Everything else uses trust-based colors
        if (node.trust_score !== undefined && node.trust_score !== null) {
            return getTrustColor(node.trust_score);
        }
        
        // Default fallback
        return CONFIG.COLORS.TRUST_UNKNOWN;
    }
    
    /**
     * Get stroke (border) color for a node
     * @param {Object} node - Node data object
     * @returns {string} Hex color code
     */
    static getStrokeColor(node) {
        // High suppression gets red border
        if (node.suppression_score && node.suppression_score > CONFIG.SUPPRESSION.HIGH_THRESHOLD) {
            return CONFIG.COLORS.SUPPRESSION_HIGH;
        }
        
        // Default white border
        return '#ffffff';
    }
    
    /**
     * Get stroke width for a node
     * @param {Object} node - Node data object
     * @returns {number} Stroke width in pixels
     */
    static getStrokeWidth(node) {
        // High suppression gets thicker border
        if (node.suppression_score && node.suppression_score > CONFIG.SUPPRESSION.HIGH_THRESHOLD) {
            return CONFIG.SIZES.STROKE_WIDTH_SUPPRESSED;
        }
        
        return CONFIG.SIZES.STROKE_WIDTH_NORMAL;
    }
    
    /**
     * Get stroke dasharray for a node
     * @param {Object} node - Node data object
     * @returns {string} SVG dasharray value
     */
    static getStrokeDasharray(node) {
        // High suppression gets dashed border
        if (node.suppression_score && node.suppression_score > CONFIG.SUPPRESSION.HIGH_THRESHOLD) {
            return CONFIG.SUPPRESSION.BORDER_DASH;
        }
        
        return 'none';
    }
    
    /**
     * Check if node should have glow effect
     * @param {Object} node - Node data object
     * @returns {boolean}
     */
    static shouldGlow(node) {
        return node.suppression_score && node.suppression_score > CONFIG.SUPPRESSION.HIGH_THRESHOLD;
    }
    
    /**
     * Get filter string for SVG glow effect
     * @param {Object} node - Node data object
     * @returns {string} SVG filter reference or 'none'
     */
    static getFilter(node) {
        if (this.shouldGlow(node)) {
            return 'url(#suppression-glow)';
        }
        return 'none';
    }
    
    /**
     * Get opacity for a node
     * @param {Object} node - Node data object
     * @returns {number} Opacity value 0-1
     */
    static getOpacity(node) {
        // Could vary based on properties
        // For now, all nodes fully opaque
        return 1.0;
    }
    
    /**
     * Get edge color based on relationship type
     * @param {Object} edge - Edge data object
     * @returns {string} Hex color code
     */
    static getEdgeColor(edge) {
        const type = edge.type || edge.relationship_type || 'MENTIONS';
        const style = CONFIG.EDGE_STYLES[type];
        return style ? style.color : CONFIG.COLORS.EDGE_MENTIONS;
    }
    
    /**
     * Get edge width based on relationship type and weight
     * @param {Object} edge - Edge data object
     * @returns {number} Edge width in pixels
     */
    static getEdgeWidth(edge) {
        const type = edge.type || edge.relationship_type || 'MENTIONS';
        const style = CONFIG.EDGE_STYLES[type];
        const baseWidth = style ? style.width : CONFIG.SIZES.EDGE_WIDTH_THIN;
        
        // Scale by weight if present
        if (edge.weight && edge.weight > 1) {
            return baseWidth * Math.sqrt(edge.weight);
        }
        
        return baseWidth;
    }
    
    /**
     * Get edge dasharray
     * @param {Object} edge - Edge data object
     * @returns {string} SVG dasharray value
     */
    static getEdgeDasharray(edge) {
        const type = edge.type || edge.relationship_type || 'MENTIONS';
        const style = CONFIG.EDGE_STYLES[type];
        return style ? style.dasharray : 'none';
    }
    
    /**
     * Get edge opacity
     * @param {Object} edge - Edge data object
     * @returns {number} Opacity value 0-1
     */
    static getEdgeOpacity(edge) {
        if (!edge) return 0.5;
        const type = edge.type || edge.relationship_type || 'MENTIONS';
        const style = CONFIG.EDGE_STYLES[type];
        return style ? style.opacity : 0.6;
    }
}
