/**
 * Aegis Insight v2.0 - Timeline Layout
 * Temporal visualization with configurable time scales and grouping
 */

import { CONFIG, getScoreColor } from '../config.js';
// Note: If this file is in js/graph/, the import path '../config.js' reaches js/config.js
import { ColorScheme } from './ColorScheme.js';

export class TimelineLayout {
    constructor(graphRenderer) {
        this.graph = graphRenderer;
        this.isActive = false;
        this.timeScale = null;
        this.yScale = null;
        this.groupBy = 'source';  // source, claim_type, topic, domain
        this.scaleType = 'month';  // day, week, month, year, decade, century
        this.timeRange = null;
        this.groups = [];
        this.originalPositions = new Map();
        
        // Restart force simulation
        if (this.graph.simulation?.restart) {
            this.graph.simulation.restart(1);
        } else if (this.graph.simulation?.simulation) {
            this.graph.simulation.simulation.alpha(1).restart();
        }
        
        // Timeline-specific elements
        this.timelineGroup = null;
        this.gridGroup = null;
        this.axisGroup = null;
        this.connectionsGroup = null;
        
        // Animation state
        this.isAnimating = false;
        this.animationFrame = null;
        
        // Auto-cleanup on filter reset or new data
        this._boundCleanup = () => this.forceDeactivate();
        document.addEventListener('aegis:filtersReset', this._boundCleanup);
        document.addEventListener('aegis:newSearch', this._boundCleanup);
        document.addEventListener('aegis:graphReload', this._boundCleanup);
        
        console.log('TimelineLayout: Initialized with auto-cleanup listeners');
    }
    
    /**
     * Destroy the timeline layout and remove event listeners
     */
    destroy() {
        this.forceDeactivate();
        document.removeEventListener('aegis:filtersReset', this._boundCleanup);
        document.removeEventListener('aegis:newSearch', this._boundCleanup);
        document.removeEventListener('aegis:graphReload', this._boundCleanup);
    }

    /**
     * Activate timeline view
     * @param {Object} options - { groupBy, scaleType }
     */
    activate(options = {}) {
        console.log("=== ACTIVATE CALLED ===", options);
        if (this.isActive) return;
        
        this.groupBy = options.groupBy || 'source';
        
        // Check for temporal data availability
        const dates = this.extractDates();
        this.fallbackMode = dates.length < 3;
        
        if (this.fallbackMode) {
            console.log(`Timeline: Fallback mode (only ${dates.length} dated nodes)`);
        }
        
        this.scaleType = options.scaleType || this.detectBestScale();
        
        // Store original positions for restoration
        this.storeOriginalPositions();
        
        // Stop force simulation
        this.graph.simulation.stop();
        
        // Create timeline layout
        this.createTimelineLayout();
        
        // Position nodes on timeline
        this.positionNodes();
        
        // Draw connections
        this.drawConnections();
        
        // Show timeline controls
        this.showControls();
        
        this.isActive = true;
    }

    /**
     * Deactivate timeline view and restore force layout
     */
    deactivate() {
        console.log("=== DEACTIVATE CALLED ===");
        // Always run cleanup even if not marked active (state can get out of sync)
        
        console.log('Deactivating timeline view...');
        
        this._cleanup();
        
        // Restart simulation with proper force
        if (this.graph.simulation?.simulation) {
            this.graph.simulation.simulation.alpha(0.8).restart();
        } else if (this.graph.simulation?.restart) {
            this.graph.simulation.restart(0.8);
        }
        
        this.isActive = false;
        this.fallbackMode = false;
        
        console.log('Timeline deactivated, force layout restored');
    }
    
    /**
     * Force deactivate - call this when graph is being reloaded
     * Can be called even if timeline isn't active (safe cleanup)
     */
    forceDeactivate() {
        console.log('Timeline: Force deactivate called');
        this._cleanup();
        this.isActive = false;
        this.fallbackMode = false;
        
        // Reset view toggle buttons to "Graph" mode
        document.querySelectorAll('.view-btn').forEach(btn => {
            if (btn.dataset.view === 'force') {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }
    
    /**
     * Internal cleanup - removes all timeline elements
     */
    _cleanup() {
        console.log("=== _cleanup called ===");
        // Remove timeline group and all children
        // Always remove by selector first (more reliable)
        d3.selectAll(".timeline-layout").remove();
        if (this.timelineGroup) {
            this.timelineGroup.remove();
            this.timelineGroup = null;
        }
        this.gridGroup = null;
        this.axisGroup = null;
        
        // Remove controls (try multiple selectors to be thorough)
        d3.select('#timeline-controls').remove();
        d3.selectAll('.timeline-controls').remove();
        
        // Also remove by class from SVG
        if (this.graph.g) {
            this.graph.g.selectAll('.timeline-layout').remove();
            this.graph.g.selectAll('.timeline-grid').remove();
            this.graph.g.selectAll('.timeline-axes').remove();
        }
        
        // Unpin all nodes
        if (this.graph.data?.nodes) {
            this.graph.data.nodes.forEach(node => {
                node.fx = null;
                node.fy = null;
            });
        }
        
        // Restore original positions if we have them
        if (this.originalPositions?.size > 0) {
            this.restoreOriginalPositions();
        }
        
        // Clear stored positions
        this.originalPositions = new Map();
        
        // Restart force simulation
        if (this.graph.simulation?.restart) {
            this.graph.simulation.restart(1);
        } else if (this.graph.simulation?.simulation) {
            this.graph.simulation.simulation.alpha(1).restart();
        }
    }

    /**
     * Toggle timeline view
     */
    toggle(options = {}) {
        if (this.isActive) {
            this.deactivate();
        } else {
            this.activate(options);
        }
    }

    /**
     * Store original node positions
     */
    storeOriginalPositions() {
        this.originalPositions.clear();
        this.graph.data.nodes.forEach(node => {
            this.originalPositions.set(node.id, { x: node.x, y: node.y });
        });
    }

    /**
     * Restore original node positions
     */
    restoreOriginalPositions() {
        console.log(`Restoring ${this.originalPositions.size} node positions...`);
        
        this.graph.data.nodes.forEach(node => {
            const original = this.originalPositions.get(node.id);
            if (original) {
                node.x = original.x;
                node.y = original.y;
                // Clear fixed positions
                node.fx = null;
                node.fy = null;
            }
        });
        
        // Update visual positions with animation
        this.graph.nodesGroup.selectAll('g.node')
            .transition()
            .duration(600)
            .attr('transform', d => `translate(${d.x},${d.y})`);
        
        // Update links after a brief delay
        setTimeout(() => {
            this.graph.linksGroup.selectAll('line')
                .attr('x1', d => d?.source?.x ?? 0)
                .attr('y1', d => d?.source?.y ?? 0)
                .attr('x2', d => d?.target?.x ?? 0)
                .attr('y2', d => d?.target?.y ?? 0);
        }, 100);
    }

    /**
     * Detect best time scale based on data range
     */
    detectBestScale() {
        const dates = this.extractDates();
        if (dates.length < 2) return 'year';
        
        const minDate = d3.min(dates);
        const maxDate = d3.max(dates);
        const rangeDays = (maxDate - minDate) / (1000 * 60 * 60 * 24);
        
        if (rangeDays < 30) return 'day';
        if (rangeDays < 180) return 'week';
        if (rangeDays < 730) return 'month';
        if (rangeDays < 3650) return 'year';
        if (rangeDays < 36500) return 'decade';
        return 'century';
    }

    /**
     * Extract dates from nodes
     */
    extractDates() {
        const dates = [];
        let nodesChecked = 0;
        let nodesWithTemporal = 0;
        
        this.graph.data.nodes.forEach(node => {
            nodesChecked++;
            if (node.temporal_data || node.metadata?.temporal_data) nodesWithTemporal++;
            
            const date = this.getNodeDate(node);
            if (date) {
                dates.push(date);
                // Log first few successful extractions for debugging
                if (dates.length <= 3) {
                    console.log(`Timeline: Found date ${date.getFullYear()} from node:`, 
                        node.text?.substring(0, 50) || node.label || node.id);
                }
            }
        });
        
        console.log(`Timeline extractDates: ${dates.length}/${nodesChecked} nodes have dates, ${nodesWithTemporal} have temporal_data property`);
        
        return dates;
    }

    /**
     * Get date from node (handles various formats and locations)
     */
    getNodeDate(node) {
        // Check both root and metadata for temporal_data
        let temporal_data = node.temporal_data || node.metadata?.temporal_data;
        // Parse if JSON string
        if (typeof temporal_data === 'string' && temporal_data.startsWith('{')) {
            try { temporal_data = JSON.parse(temporal_data); } catch(e) { temporal_data = null; }
        }
        // 1. Try structured temporal_data (various formats)
        if (temporal_data) {
            // Format 1: { absolute_dates: [{date: "1934"}] }
            if (temporal_data.absolute_dates?.length > 0) {
                const dateStr = temporal_data.absolute_dates[0].date || 
                               temporal_data.absolute_dates[0];
                const parsed = this.parseDate(dateStr);
                if (parsed) return parsed;
            }
            
            // Format 2: { dates: ["1934-10-22"] }
            if (temporal_data.dates?.length > 0) {
                const parsed = this.parseDate(temporal_data.dates[0]);
                if (parsed) return parsed;
            }
            
            // Format 3: { year: 1934 } or { date: "1934" }
            if (temporal_data.year) {
                return new Date(temporal_data.year, 0, 1);
            }
            if (temporal_data.date) {
                const parsed = this.parseDate(temporal_data.date);
                if (parsed) return parsed;
            }
            
            // Format 4: temporal_data is a string directly
            if (typeof temporal_data === 'string') {
                const parsed = this.parseDate(temporal_data);
                if (parsed) return parsed;
            }
        }
        
        // 2. Try direct date properties
        if (node.date) {
            const parsed = this.parseDate(node.date);
            if (parsed) return parsed;
        }
        
        if (node.year) {
            return new Date(node.year, 0, 1);
        }
        
        if (node.publication_date) {
            const parsed = this.parseDate(node.publication_date);
            if (parsed) return parsed;
        }
        
        // 3. Try to extract year from source filename
        // e.g., "FBI_vault_1934.pdf" or "document_1930s.jsonl"
        const source = node.source || node.source_file || '';
        const sourceYearMatch = source.match(/[_\-\s](\d{4})[_\-\s.]/);
        if (sourceYearMatch) {
            const year = parseInt(sourceYearMatch[1]);
            if (year >= 1800 && year <= 2100) {
                return new Date(year, 0, 1);
            }
        }
        
        // 4. Try to extract year from claim text
        // Look for 4-digit years in reasonable range (1800-2100)
        const text = node.text || node.claim_text || node.label || '';
        const textYearMatch = text.match(/\b(1[89]\d{2}|20[0-2]\d)\b/);
        if (textYearMatch) {
            const year = parseInt(textYearMatch[1]);
            // Sanity check - don't use if it looks like a number (e.g., "1000 soldiers")
            const context = text.substring(
                Math.max(0, text.indexOf(textYearMatch[1]) - 10),
                text.indexOf(textYearMatch[1]) + 15
            );
            // Avoid matching things like "$1934" or "1934 soldiers"
            if (!context.match(/\$|soldiers|troops|men|people|dollars|pounds/i)) {
                return new Date(year, 0, 1);
            }
        }
        
        // 5. Try created_at as last resort
        if (node.created_at) {
            return new Date(node.created_at);
        }
        
        return null;
    }

    /**
     * Parse date string (handles BCE, relative dates, etc.)
     */
    parseDate(dateStr) {
        if (!dateStr) return null;
        
        // Handle BCE dates
        if (dateStr.includes('BCE') || dateStr.includes('BC')) {
            const year = parseInt(dateStr.match(/-?\d+/)?.[0] || '0');
            // Use negative year for BCE
            return new Date(-year, 0, 1);
        }
        
        // Handle year-only
        if (/^\d{4}$/.test(dateStr)) {
            return new Date(parseInt(dateStr), 0, 1);
        }
        
        // Standard date parsing
        const parsed = new Date(dateStr);
        return isNaN(parsed.getTime()) ? null : parsed;
    }

    /**
     * Create timeline layout structure
     * IMPORTANT: Grid is inserted BEFORE (behind) the nodes group
     */
    createTimelineLayout() {
        console.log('Creating timeline layout...');
        const g = this.graph.g;
        const margin = CONFIG.TIMELINE.MARGIN;
        const width = this.graph.width - margin.left - margin.right;
        const height = this.graph.height - margin.top - margin.bottom;
        
        console.log(`Timeline dimensions: ${width}x${height}, margin: ${JSON.stringify(margin)}`);
        
        // Remove any existing timeline elements
        g.selectAll('.timeline-layout').remove();
        
        // Create timeline container - INSERT BEFORE links/nodes so it's behind them
        // Use insert() instead of append() to place at beginning
        this.timelineGroup = g.insert('g', ':first-child')
            .attr('class', 'timeline-layout')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        // Create sub-groups
        this.gridGroup = this.timelineGroup.append('g').attr('class', 'timeline-grid');
        this.axisGroup = this.timelineGroup.append('g').attr('class', 'timeline-axes');
        
        console.log('Timeline groups created, building scales...');
        
        // Build scales
        this.buildScales(width, height);
        
        // Draw grid (transparent, behind nodes)
        this.drawGrid(width, height);
        
        // Draw axes
        this.drawAxes(width, height);
        
        console.log('Timeline layout complete');
    }

    /**
     * Build time and group scales
     */
    buildScales(width, height) {
        const dates = this.extractDates().filter(d => d !== null);
        
        console.log(`Timeline buildScales: Found ${dates.length} dated nodes out of ${this.graph.data.nodes.length}`);
        
        // Time scale (X-axis)
        if (dates.length > 0) {
            const minDate = d3.min(dates);
            const maxDate = d3.max(dates);
            
            console.log(`Timeline date range: ${minDate.getFullYear()} - ${maxDate.getFullYear()}`);
            
            // Add padding to date range
            const rangePadding = (maxDate - minDate) * 0.05 || (365 * 24 * 60 * 60 * 1000); // 1 year if same date
            
            this.timeScale = d3.scaleTime()
                .domain([new Date(minDate.getTime() - rangePadding), 
                         new Date(maxDate.getTime() + rangePadding)])
                .range([0, width]);
            
            this.timeRange = { min: minDate, max: maxDate };
        } else {
            // No dates found - try to guess from content
            console.log('Timeline: No dates found, attempting to extract from content...');
            
            // Try to find any year mentions in claim text
            const yearHints = [];
            this.graph.data.nodes.forEach(node => {
                const text = node.text || node.claim_text || node.label || '';
                const matches = text.match(/\b(1[89]\d{2}|20[0-2]\d)\b/g);
                if (matches) {
                    matches.forEach(m => yearHints.push(parseInt(m)));
                }
            });
            
            if (yearHints.length > 0) {
                const minYear = Math.min(...yearHints);
                const maxYear = Math.max(...yearHints);
                console.log(`Timeline: Extracted year range from text: ${minYear} - ${maxYear}`);
                
                this.timeScale = d3.scaleTime()
                    .domain([new Date(minYear, 0, 1), new Date(maxYear, 11, 31)])
                    .range([0, width]);
                
                this.timeRange = { min: new Date(minYear, 0, 1), max: new Date(maxYear, 11, 31) };
            } else {
                // Final fallback - use a generic historical range
                console.log('Timeline: No year hints found, using generic range');
                this.timeScale = d3.scaleTime()
                    .domain([new Date(1900, 0, 1), new Date(2000, 11, 31)])
                    .range([0, width]);
                
                this.timeRange = null;
            }
        }
        
        // Build groups for Y-axis
        this.buildGroups();
        
        // Y scale
        const groupCount = Math.max(this.groups.length, 1);
        const rowHeight = Math.min(CONFIG.TIMELINE.ROW_HEIGHT, height / groupCount);
        
        this.yScale = d3.scaleBand()
            .domain(this.groups.map(g => g.id))
            .range([0, height])
            .padding(0.1);
    }

    /**
     * Build groups for Y-axis based on groupBy setting
     */
    buildGroups() {
        this.groups = [];
        const groupMap = new Map();
        
        this.graph.data.nodes.forEach(node => {
            let groupId, groupLabel;
            
            switch (this.groupBy) {
                case 'source':
                    groupId = node.source_file || node.source || 'Unknown';
                    groupLabel = this.truncateLabel(groupId.split('/').pop(), 30);
                    break;
                    
                case 'claim_type':
                    groupId = node.claim_type || 'UNKNOWN';
                    groupLabel = groupId;
                    break;
                    
                case 'topic':
                    groupId = node.topic || node.category || 'General';
                    groupLabel = this.truncateLabel(groupId, 25);
                    break;
                    
                case 'domain':
                    groupId = node.authority_domain || node.domain || 'Unknown';
                    groupLabel = groupId;
                    break;
                    
                default:
                    groupId = 'All';
                    groupLabel = 'All Claims';
            }
            
            if (!groupMap.has(groupId)) {
                groupMap.set(groupId, { id: groupId, label: groupLabel, count: 0 });
            }
            groupMap.get(groupId).count++;
            
            // Store group on node for positioning
            node._timelineGroup = groupId;
        });
        
        // Sort groups by count (most claims first)
        this.groups = Array.from(groupMap.values())
            .sort((a, b) => b.count - a.count);
        
        // Limit to reasonable number
        if (this.groups.length > 20) {
            const others = this.groups.slice(20);
            const otherCount = others.reduce((sum, g) => sum + g.count, 0);
            this.groups = this.groups.slice(0, 20);
            this.groups.push({ id: '_other', label: `Other (${others.length} groups)`, count: otherCount });
            
            // Reassign nodes
            this.graph.data.nodes.forEach(node => {
                if (!this.groups.find(g => g.id === node._timelineGroup)) {
                    node._timelineGroup = '_other';
                }
            });
        }
    }

    /**
     * Draw Erlenmeyer flask-style graduation marks
     * Tick marks at TOP and BOTTOM of viewport only (like flask graduations)
     * NOT a full grid overlay - leaves the main graph area clear
     */
    drawGrid(width, height) {
        console.log(`Drawing timeline graduations: ${width}x${height}`);
        
        // Clear existing
        this.gridGroup.selectAll('*').remove();
        
        // Colors
        const tickColor = 'rgba(0, 204, 255, 0.7)';
        const minorTickColor = 'rgba(0, 204, 255, 0.3)';
        const labelColor = 'rgba(0, 204, 255, 0.9)';
        const barBg = 'rgba(0, 20, 40, 0.8)';
        
        // Tick dimensions
        const majorTickHeight = 12;
        const minorTickHeight = 6;
        const barHeight = 30;
        
        // === TOP GRADUATION BAR ===
        this.gridGroup.append('rect')
            .attr('x', 0)
            .attr('y', 0)
            .attr('width', width)
            .attr('height', barHeight)
            .attr('fill', barBg);
        
        // Top baseline
        this.gridGroup.append('line')
            .attr('x1', 0).attr('x2', width)
            .attr('y1', barHeight).attr('y2', barHeight)
            .attr('stroke', tickColor)
            .attr('stroke-width', 1.5);
        
        // === BOTTOM GRADUATION BAR ===
        this.gridGroup.append('rect')
            .attr('x', 0)
            .attr('y', height - barHeight)
            .attr('width', width)
            .attr('height', barHeight)
            .attr('fill', barBg);
        
        // Bottom baseline
        this.gridGroup.append('line')
            .attr('x1', 0).attr('x2', width)
            .attr('y1', height - barHeight).attr('y2', height - barHeight)
            .attr('stroke', tickColor)
            .attr('stroke-width', 1.5);
        
        // === TIME TICKS ===
        if (this.timeScale) {
            const scaleConfig = CONFIG.TIMELINE.SCALES[this.scaleType] || { ticks: 10 };
            const majorTicks = this.timeScale.ticks(scaleConfig.ticks);
            const minorTicks = this.timeScale.ticks(scaleConfig.ticks * 4);
            
            console.log(`Timeline: ${majorTicks.length} major, ${minorTicks.length} minor ticks`);
            
            // Minor ticks
            minorTicks.forEach(tick => {
                const x = this.timeScale(tick);
                
                // Top tick (down from bar)
                this.gridGroup.append('line')
                    .attr('x1', x).attr('x2', x)
                    .attr('y1', barHeight - minorTickHeight).attr('y2', barHeight)
                    .attr('stroke', minorTickColor)
                    .attr('stroke-width', 1);
                
                // Bottom tick (up from bar)
                this.gridGroup.append('line')
                    .attr('x1', x).attr('x2', x)
                    .attr('y1', height - barHeight).attr('y2', height - barHeight + minorTickHeight)
                    .attr('stroke', minorTickColor)
                    .attr('stroke-width', 1);
            });
            
            // Major ticks with labels
            majorTicks.forEach(tick => {
                const x = this.timeScale(tick);
                const label = this.formatTickLabel(tick);
                
                // Top major tick
                this.gridGroup.append('line')
                    .attr('x1', x).attr('x2', x)
                    .attr('y1', barHeight - majorTickHeight).attr('y2', barHeight)
                    .attr('stroke', tickColor)
                    .attr('stroke-width', 2);
                
                // Top label
                this.gridGroup.append('text')
                    .attr('x', x)
                    .attr('y', 12)
                    .attr('text-anchor', 'middle')
                    .attr('fill', labelColor)
                    .attr('font-size', '10px')
                    .attr('font-family', 'monospace')
                    .text(label);
                
                // Bottom major tick
                this.gridGroup.append('line')
                    .attr('x1', x).attr('x2', x)
                    .attr('y1', height - barHeight).attr('y2', height - barHeight + majorTickHeight)
                    .attr('stroke', tickColor)
                    .attr('stroke-width', 2);
                
                // Bottom label
                this.gridGroup.append('text')
                    .attr('x', x)
                    .attr('y', height - 8)
                    .attr('text-anchor', 'middle')
                    .attr('fill', labelColor)
                    .attr('font-size', '10px')
                    .attr('font-family', 'monospace')
                    .text(label);
            });
        }
        
        // === DIRECTION INDICATORS ===
        this.gridGroup.append('text')
            .attr('x', 8)
            .attr('y', 20)
            .attr('fill', labelColor)
            .attr('font-size', '11px')
            .attr('font-weight', 'bold')
            .text('◀ EARLIER');
        
        this.gridGroup.append('text')
            .attr('x', width - 8)
            .attr('y', 20)
            .attr('text-anchor', 'end')
            .attr('fill', labelColor)
            .attr('font-size', '11px')
            .attr('font-weight', 'bold')
            .text('LATER ▶');
        
        // === MODE LABEL (center) ===
        this.gridGroup.append('text')
            .attr('x', width / 2)
            .attr('y', 20)
            .attr('text-anchor', 'middle')
            .attr('fill', '#00ccff')
            .attr('font-size', '11px')
            .attr('font-family', 'monospace')
            .attr('font-weight', 'bold')
            .text('⏱ TIMELINE');
        
        // === FALLBACK MESSAGE (centered in graph area) ===
        if (this.fallbackMode) {
            const msgY = height / 2;
            
            this.gridGroup.append('rect')
                .attr('x', width/2 - 190)
                .attr('y', msgY - 35)
                .attr('width', 380)
                .attr('height', 70)
                .attr('rx', 8)
                .attr('fill', 'rgba(255, 152, 0, 0.12)')
                .attr('stroke', 'rgba(255, 152, 0, 0.5)')
                .attr('stroke-width', 2);
            
            this.gridGroup.append('text')
                .attr('x', width/2)
                .attr('y', msgY - 8)
                .attr('text-anchor', 'middle')
                .attr('fill', '#FF9800')
                .attr('font-size', '14px')
                .attr('font-weight', 'bold')
                .text('⚠ Limited Temporal Data');
            
            this.gridGroup.append('text')
                .attr('x', width/2)
                .attr('y', msgY + 15)
                .attr('text-anchor', 'middle')
                .attr('fill', 'rgba(255, 152, 0, 0.8)')
                .attr('font-size', '12px')
                .text('Claims spread by source (insufficient date information)');
        }
    }
    
    /**
     * Format tick label based on scale type
     */
    formatTickLabel(date) {
        if (!date || !(date instanceof Date)) return '';
        
        switch (this.scaleType) {
            case 'day': return d3.timeFormat('%b %d')(date);
            case 'week': return d3.timeFormat('%b %d')(date);
            case 'month': return d3.timeFormat('%b %Y')(date);
            case 'year': return d3.timeFormat('%Y')(date);
            case 'decade': return d3.timeFormat('%Y')(date) + 's';
            case 'century': return Math.floor(date.getFullYear() / 100) + 1 + 'c';
            default: return d3.timeFormat('%Y')(date);
        }
    }
    /**
     * Draw axes
     */
    drawAxes(width, height) {
        // X-axis (time)
        const scaleConfig = CONFIG.TIMELINE.SCALES[this.scaleType] || { ticks: 10, format: "YYYY" };
        
        const xAxis = d3.axisBottom(this.timeScale)
            .ticks(scaleConfig.ticks)
            .tickFormat(d => this.formatDate(d, scaleConfig.format));
        
        this.axisGroup.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0,${height})`)
            .call(xAxis)
            .selectAll('text')
            .attr('fill', CONFIG.COLORS.TEXT_SECONDARY)
            .attr('font-size', '11px');
        
        this.axisGroup.selectAll('.x-axis path, .x-axis line')
            .attr('stroke', CONFIG.TIMELINE.AXIS_COLOR);
        
        // Y-axis (groups)
        const yAxis = d3.axisLeft(this.yScale)
            .tickFormat(d => {
                const group = this.groups.find(g => g.id === d);
                return group ? group.label : d;
            });
        
        this.axisGroup.append('g')
            .attr('class', 'y-axis')
            .call(yAxis)
            .selectAll('text')
            .attr('fill', CONFIG.COLORS.TEXT_SECONDARY)
            .attr('font-size', '11px');
        
        this.axisGroup.selectAll('.y-axis path, .y-axis line')
            .attr('stroke', CONFIG.TIMELINE.AXIS_COLOR);
        
        // Axis labels
        this.axisGroup.append('text')
            .attr('class', 'x-axis-label')
            .attr('x', width / 2)
            .attr('y', height + 45)
            .attr('text-anchor', 'middle')
            .attr('fill', CONFIG.COLORS.TEXT_PRIMARY)
            .attr('font-size', '13px')
            .text('Time');
        
        this.axisGroup.append('text')
            .attr('class', 'y-axis-label')
            .attr('transform', 'rotate(-90)')
            .attr('x', -height / 2)
            .attr('y', -CONFIG.TIMELINE.MARGIN.left + 20)
            .attr('text-anchor', 'middle')
            .attr('fill', CONFIG.COLORS.TEXT_PRIMARY)
            .attr('font-size', '13px')
            .text(CONFIG.TIMELINE.GROUP_BY[this.groupBy] || this.groupBy);
    }

    /**
     * Format date based on scale
     */
    formatDate(date, format) {
        if (!date || isNaN(date.getTime())) return '';
        
        const year = date.getFullYear();
        const month = date.toLocaleString('en', { month: 'short' });
        const day = date.getDate();
        
        // Handle BCE dates
        if (year < 0) {
            return `${Math.abs(year)} BCE`;
        }
        
        switch (format) {
            case 'MMM D':
                return `${month} ${day}`;
            case 'MMM YYYY':
                return `${month} ${year}`;
            case 'YYYY':
                return `${year}`;
            default:
                return `${month} ${day}, ${year}`;
        }
    }

    /**
     * Position nodes on timeline
     */
    positionNodes() {
        const margin = CONFIG.TIMELINE.MARGIN;
        const undatedNodes = [];
        
        this.graph.data.nodes.forEach(node => {
            const date = this.getNodeDate(node);
            const groupId = node._timelineGroup || '_other';
            
            if (date && this.yScale(groupId) !== undefined) {
                // Calculate timeline position
                node.x = margin.left + this.timeScale(date);
                node.y = margin.top + this.yScale(groupId) + this.yScale.bandwidth() / 2;
                
                // Jitter overlapping nodes
                node.x += (Math.random() - 0.5) * 10;
                node.y += (Math.random() - 0.5) * (this.yScale.bandwidth() * 0.6);
            } else {
                undatedNodes.push(node);
            }
        });
        
        // Stack undated nodes on the right
        if (undatedNodes.length > 0) {
            const undatedX = this.graph.width - 60;
            undatedNodes.forEach((node, i) => {
                node.x = undatedX + (Math.floor(i / 15) * 30);
                node.y = margin.top + 30 + (i % 15) * 25;
            });
        }
        
        // Animate to new positions
        this.graph.nodesGroup.selectAll('g.node')
            .transition()
            .duration(750)
            .ease(d3.easeCubicInOut)
            .attr('transform', d => `translate(${d.x},${d.y})`);
        
        // Show undated indicator
        if (undatedNodes.length > 0) {
            this.showUndatedIndicator(undatedNodes.length);
        }
    }

    /**
     * Draw connections in timeline view
     */
    drawConnections() {
        // Update edge positions
        this.graph.linksGroup.selectAll('line')
            .transition()
            .duration(750)
            .ease(d3.easeCubicInOut)
            .attr('x1', d => d?.source?.x ?? 0)
            .attr('y1', d => d?.source?.y ?? 0)
            .attr('x2', d => d?.target?.x ?? 0)
            .attr('y2', d => d?.target?.y ?? 0)
            .attr('stroke-opacity', 0.3);  // Dim edges in timeline view
    }

    /**
     * Show indicator for undated nodes
     */
    showUndatedIndicator(count) {
        this.timelineGroup.append('text')
            .attr('class', 'undated-indicator')
            .attr('x', this.graph.width - CONFIG.TIMELINE.MARGIN.left - 60)
            .attr('y', 15)
            .attr('text-anchor', 'middle')
            .attr('fill', CONFIG.COLORS.TEXT_SECONDARY)
            .attr('font-size', '11px')
            .text(`${count} undated`);
    }

    /**
     * Show timeline controls
     */
    showControls() {
        console.log("=== showControls called ===");
        try {
        // Remove existing controls
        d3.select('#timeline-controls').remove();
        
        const controls = d3.select(this.graph.container)
            .append('div')
            .attr('id', 'timeline-controls')
            .attr('class', 'timeline-controls')
            .style('position', 'absolute')
            .style('bottom', '80px')
            .style('left', '50%')
            .style('transform', 'translateX(-50%)')
            .style('background', 'rgba(28, 36, 52, 0.95)')
            .style('border', '1px solid #2a3a5a')
            .style('border-radius', '8px')
            .style('padding', '12px 20px')
            .style('display', 'flex')
            .style('flex-direction', 'row')
            .style('width', 'auto')
            .style('max-width', '600px')
            .style('height', 'auto')
            .style('max-height', '60px')
            .style('gap', '20px')
            .style('align-items', 'center')
            .style('z-index', '100');
        
        // Time scale selector
        const scaleGroup = controls.append('div')
            .style('display', 'flex')
            .style('flex-direction', 'row')
            .style('width', 'auto')
            .style('max-width', '600px')
            .style('height', 'auto')
            .style('max-height', '60px')
            .style('align-items', 'center')
            .style('gap', '8px');
        
        scaleGroup.append('label')
            .style('color', CONFIG.COLORS.TEXT_SECONDARY)
            .style('font-size', '12px')
            .text('Scale:');
        
        const scaleSelect = scaleGroup.append('select')
            .attr('id', 'timeline-scale-select')
            .style('background', 'rgba(0,0,0,0.3)')
            .style('border', '1px solid #2a3a5a')
            .style('color', CONFIG.COLORS.TEXT_PRIMARY)
            .style('padding', '4px 8px')
            .style('border-radius', '4px')
            .on('change', () => {
                this.scaleType = scaleSelect.property('value');
                this.refresh();
            });
        
        Object.keys(CONFIG.TIMELINE.SCALES).forEach(scale => {
            scaleSelect.append('option')
                .attr('value', scale)
                .property('selected', scale === this.scaleType)
                .text(scale.charAt(0).toUpperCase() + scale.slice(1));
        });
        
        // Group by selector
        const groupGroup = controls.append('div')
            .style('display', 'flex')
            .style('flex-direction', 'row')
            .style('width', 'auto')
            .style('max-width', '600px')
            .style('height', 'auto')
            .style('max-height', '60px')
            .style('align-items', 'center')
            .style('gap', '8px');
        
        groupGroup.append('label')
            .style('color', CONFIG.COLORS.TEXT_SECONDARY)
            .style('font-size', '12px')
            .text('Group by:');
        
        const groupSelect = groupGroup.append('select')
            .attr('id', 'timeline-group-select')
            .style('background', 'rgba(0,0,0,0.3)')
            .style('border', '1px solid #2a3a5a')
            .style('color', CONFIG.COLORS.TEXT_PRIMARY)
            .style('padding', '4px 8px')
            .style('border-radius', '4px')
            .on('change', () => {
                this.groupBy = groupSelect.property('value');
                this.refresh();
            });
        
        Object.entries(CONFIG.TIMELINE.GROUP_BY).forEach(([value, label]) => {
            groupSelect.append('option')
                .attr('value', value)
                .property('selected', value === this.groupBy)
                .text(label);
        });
        
        // Play animation button
        controls.append('button')
            .attr('id', 'timeline-play-btn')
            .style('background', CONFIG.COLORS.ACCENT)
            .style('border', 'none')
            .style('color', '#0a0e17')
            .style('padding', '6px 12px')
            .style('border-radius', '4px')
            .style('cursor', 'pointer')
            .style('font-weight', 'bold')
            .text('▶ Play')
            .on('click', () => this.toggleAnimation());
        
        // Reset Layout button
        controls.append('button')
            .style('background', 'rgba(255, 152, 0, 0.2)')
            .style('border', '1px solid rgba(255, 152, 0, 0.5)')
            .style('color', '#FF9800')
            .style('padding', '6px 12px')
            .style('border-radius', '4px')
            .style('cursor', 'pointer')
            .text('Reset Layout')
            .on('click', () => {
                // Unpin and let simulation re-settle
                this.graph.data.nodes.forEach(n => { n.fx = null; n.fy = null; });
                if (this.graph.simulation?.simulation) {
                    this.graph.simulation.simulation.alpha(0.5).restart();
                }
            });
        
        // Exit timeline button
        controls.append('button')
            .style('background', 'rgba(255,56,96,0.2)')
            .style('border', '1px solid rgba(255,56,96,0.5)')
            .style('color', '#FF3860')
            .style('padding', '6px 12px')
            .style('border-radius', '4px')
            .style('cursor', 'pointer')
            .text('Exit Timeline')
            .on('click', () => this.deactivate());
        } catch(e) { console.error('showControls error:', e); }
    }

    /**
     * Refresh timeline with current settings
     */
    refresh() {
        console.log('Refreshing timeline layout...');
        
        // Remove existing timeline elements
        // Always remove by selector first (more reliable)
        d3.selectAll(".timeline-layout").remove();
        if (this.timelineGroup) {
            this.timelineGroup.remove();
            this.timelineGroup = null;
        }
        
        // Unpin all nodes before re-positioning
        this.graph.data.nodes.forEach(node => {
            node.fx = null;
            node.fy = null;
        });
        
        // Recreate layout
        this.createTimelineLayout();
        this.positionNodes();
        this.drawConnections();
        
        console.log('Timeline refreshed');
    }

    /**
     * Toggle animation playback
     */
    toggleAnimation() {
        if (this.isAnimating) {
            this.stopAnimation();
        } else {
            this.startAnimation();
        }
    }

    /**
     * Start timeline animation
     */
    startAnimation() {
        if (!this.timeRange) return;
        
        this.isAnimating = true;
        d3.select('#timeline-play-btn').text('⏸ Pause');
        
        const nodes = this.graph.data.nodes;
        const startTime = this.timeRange.min.getTime();
        const endTime = this.timeRange.max.getTime();
        const duration = (endTime - startTime) / CONFIG.TIMELINE.ANIMATION_SPEED;
        
        // Hide all nodes initially
        this.graph.nodesGroup.selectAll('g.node')
            .style('opacity', 0);
        
        // Animate nodes appearing over time
        let currentTime = startTime;
        const step = (endTime - startTime) / 100;  // 100 steps
        
        const animate = () => {
            if (!this.isAnimating || currentTime > endTime) {
                this.stopAnimation();
                return;
            }
            
            // Show nodes up to current time
            this.graph.nodesGroup.selectAll('g.node')
                .filter(d => {
                    const date = this.getNodeDate(d);
                    return date && date.getTime() <= currentTime;
                })
                .transition()
                .duration(50)
                .style('opacity', 1);
            
            // Update time indicator
            this.updateTimeIndicator(new Date(currentTime));
            
            currentTime += step;
            this.animationFrame = requestAnimationFrame(animate);
        };
        
        animate();
    }

    /**
     * Stop timeline animation
     */
    stopAnimation() {
        this.isAnimating = false;
        d3.select('#timeline-play-btn').text('▶ Play');
        
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
        
        // Show all nodes
        this.graph.nodesGroup.selectAll('g.node')
            .style('opacity', 1);
        
        // Remove time indicator
        this.timelineGroup?.select('.time-indicator').remove();
    }

    /**
     * Update time indicator during animation
     */
    updateTimeIndicator(date) {
        const x = this.timeScale(date);
        
        let indicator = this.timelineGroup.select('.time-indicator');
        
        if (indicator.empty()) {
            indicator = this.timelineGroup.append('line')
                .attr('class', 'time-indicator')
                .attr('y1', 0)
                .attr('y2', this.graph.height - CONFIG.TIMELINE.MARGIN.top - CONFIG.TIMELINE.MARGIN.bottom)
                .attr('stroke', CONFIG.COLORS.ACCENT)
                .attr('stroke-width', 2)
                .attr('stroke-dasharray', '4,4');
        }
        
        indicator.attr('x1', x).attr('x2', x);
    }

    /**
     * Truncate label
     */
    truncateLabel(text, maxLength) {
        if (!text) return '';
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength - 3) + '...';
    }

    /**
     * Check if timeline is active
     */
    getIsActive() {
        return this.isActive;
    }
}
