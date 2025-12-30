/**
 * Aegis Insight v2.0 - Node Shapes
 * Renders different SVG shapes based on node type
 */

import { CONFIG } from '../config.js';
import { ColorScheme } from './ColorScheme.js';

export class NodeShapes {
    /**
     * Normalize node type to lowercase
     * @param {string} type - Original type
     * @returns {string} Normalized type
     */
    static normalizeType(type) {
        if (!type) return 'entity';

        const normalized = type.toLowerCase();

        // Map common variations
        const typeMap = {
            'claim': 'claim',
            'entity': 'entity',
            'source': 'source',
            'document': 'source',
            'chunk': 'entity',
            'umbrella': 'umbrella',
            'meta': 'meta',
            'metaclaim': 'meta'
        };

        return typeMap[normalized] || 'entity';
    }

    /**
     * Render a node with appropriate shape based on type
     * @param {d3.Selection} selection - D3 selection of <g> elements
     */
    static renderNode(selection) {
        selection.each(function(d) {
            const g = d3.select(this);
            g.selectAll('*').remove(); // Clear previous content

            // Normalize type
            const nodeType = NodeShapes.normalizeType(d.type);

            const color = ColorScheme.getNodeColor(d);
            const strokeColor = ColorScheme.getStrokeColor(d);
            const strokeWidth = ColorScheme.getStrokeWidth(d);
            const strokeDasharray = ColorScheme.getStrokeDasharray(d);
            const filter = ColorScheme.getFilter(d);

            // Determine size
            const radius = d.importance ? CONFIG.SIZES.NODE_RADIUS_LARGE : CONFIG.SIZES.NODE_RADIUS_SMALL;

            switch(nodeType) {
                case 'entity':
                case 'claim':
                    // Circle (filled)
                    g.append('circle')
                        .attr('r', radius)
                        .attr('fill', color)
                        .attr('stroke', strokeColor)
                        .attr('stroke-width', strokeWidth)
                        .attr('stroke-dasharray', strokeDasharray)
                        .attr('filter', filter);
                    break;

                case 'source':
                case 'document':
                    // Square
                    const squareSize = radius * 1.75; // ~14px for radius=8
                    g.append('rect')
                        .attr('width', squareSize)
                        .attr('height', squareSize)
                        .attr('x', -squareSize / 2)
                        .attr('y', -squareSize / 2)
                        .attr('fill', color)
                        .attr('stroke', strokeColor)
                        .attr('stroke-width', strokeWidth)
                        .attr('rx', 2) // Slight rounding
                        .attr('ry', 2);
                    break;

                case 'umbrella':
                    // Hollow circle (container)
                    const containerRadius = d.radius || CONFIG.SIZES.NODE_RADIUS_CONTAINER;
                    g.append('circle')
                        .attr('r', containerRadius)
                        .attr('fill', 'none')
                        .attr('stroke', color)
                        .attr('stroke-width', 2)
                        .attr('stroke-dasharray', '8,4')
                        .attr('opacity', 0.5);
                    break;

                case 'meta':
                    // Hexagon
                    const hexPath = NodeShapes.createHexagonPath(radius);
                    g.append('path')
                        .attr('d', hexPath)
                        .attr('fill', color)
                        .attr('stroke', strokeColor)
                        .attr('stroke-width', strokeWidth);
                    break;

                default:
                    // Default to circle
                    g.append('circle')
                        .attr('r', radius)
                        .attr('fill', color)
                        .attr('stroke', strokeColor)
                        .attr('stroke-width', strokeWidth);
            }

            // Add label (now with smaller text for claims)
            NodeShapes.addLabel(g, d, nodeType);
        });
    }

    /**
     * Add text label to node
     * @param {d3.Selection} g - Group selection
     * @param {Object} d - Node data
     * @param {string} nodeType - Normalized node type
     */
    static addLabel(g, d, nodeType) {
        const isContainer = nodeType === 'umbrella';
        const containerRadius = d.radius || CONFIG.SIZES.NODE_RADIUS_CONTAINER;

        const dy = isContainer ? -containerRadius - 10 : CONFIG.SIZES.NODE_RADIUS_SMALL + 12;

        // Determine font size based on node type
        let fontSize;
        if (isContainer) {
            fontSize = CONFIG.SIZES.TEXT_CONTAINER;
        } else if (nodeType === 'claim') {
            fontSize = 9; // Smaller for claims
        } else {
            fontSize = CONFIG.SIZES.TEXT_BASE; // 11px for entities
        }

        // Truncate label intelligently
        const maxLength = nodeType === 'claim' ? 50 : 30;
        const truncatedLabel = NodeShapes.truncateLabel(d.label || d.name || d.id, maxLength);

        g.append('text')
            .attr('dy', dy)
            .attr('text-anchor', 'middle')
            .style('font-size', `${fontSize}px`)
            .style('fill', CONFIG.COLORS.TEXT_PRIMARY)
            .style('pointer-events', 'none')
            .style('user-select', 'none')
            .text(truncatedLabel);
    }

    /**
     * Truncate label to max length
     * @param {string} text - Label text
     * @param {number} maxLength - Maximum length
     * @returns {string} Truncated text
     */
    static truncateLabel(text, maxLength = 50) {
        if (!text) return '';
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength - 3) + '...';
    }

    /**
     * Create hexagon path
     * @param {number} radius - Hexagon radius
     * @returns {string} SVG path string
     */
    static createHexagonPath(radius) {
        const angle = Math.PI / 3; // 60 degrees
        const points = [];

        for (let i = 0; i < 6; i++) {
            const x = radius * Math.cos(angle * i - Math.PI / 2);
            const y = radius * Math.sin(angle * i - Math.PI / 2);
            points.push(`${x},${y}`);
        }

        return `M${points.join('L')}Z`;
    }

    /**
     * Update label visibility based on node velocity (level-of-detail)
     * @param {d3.Selection} selection - Selection of node groups
     */
    static updateLabelVisibility(selection) {
        selection.each(function(d) {
            const speed = Math.sqrt((d.vx || 0) ** 2 + (d.vy || 0) ** 2);
            const isMovingFast = speed > CONFIG.PERFORMANCE.LOD_SPEED_THRESHOLD;

            d3.select(this).select('text')
                .style('opacity', isMovingFast ? 0 : 1);
        });
    }

    /**
     * Get node size (for collision detection)
     * @param {Object} node - Node data
     * @returns {number} Radius in pixels
     */
    static getNodeSize(node) {
        const nodeType = NodeShapes.normalizeType(node.type);

        if (nodeType === 'umbrella') {
            return node.radius || CONFIG.SIZES.NODE_RADIUS_CONTAINER;
        }
        return node.importance ? CONFIG.SIZES.NODE_RADIUS_LARGE : CONFIG.SIZES.NODE_RADIUS_SMALL;
    }
}