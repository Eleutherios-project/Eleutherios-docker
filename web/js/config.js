/**
 * Aegis Insight v2.0 - Configuration
 * Central configuration for colors, shapes, sizes, and API endpoints
 * 
 * UPDATED: Added Detection Modes, Timeline, Coordination, Anomaly configs
 */
import { AdminPanel } from './admin/AdminPanel.js';

let adminPanel = null;

export const CONFIG = {
    // API Configuration
    API_URL: 'http://localhost:8001',
    
    // Color Palette v2.0
    COLORS: {
        // Trust Score Colors
        TRUST_HIGH: '#4CAF50',      // Green (0.7-1.0)
        TRUST_MEDIUM: '#FF9800',    // Orange (0.5-0.7) - REPLACES pink!
        TRUST_LOW: '#F44336',       // Red (0.0-0.5)
        TRUST_UNKNOWN: '#9E9E9E',   // Gray (no score)
        
        // Claim Position Colors
        POSITION_SUPPORT: '#2196F3',   // Blue
        POSITION_OPPOSE: '#E91E63',    // Deep pink
        POSITION_NEUTRAL: '#00BCD4',   // Cyan
        POSITION_CONTEXT: '#9E9E9E',   // Gray
        
        // Edge/Relationship Colors
        EDGE_MENTIONS: '#a0a0a0',      // Gray
        EDGE_SUPPORTS: '#4CAF50',      // Green
        EDGE_CONTRADICTS: '#E91E63',   // Pink
        EDGE_CITES: '#00ccff',         // Cyan
        EDGE_CONTAINS: '#607D8B',      // Blue-gray
        
        // Special Indicators
        SUPPRESSION_HIGH: '#FF3860',   // Red (for border/glow)
        SUPPRESSION_MEDIUM: '#ff8800', // Orange
        SUPPRESSION_LOW: '#ffcc00',    // Yellow
        
        // Coordination Detection
        COORDINATION_PRIMARY: '#ff9800',
        COORDINATION_CLUSTER: 'rgba(255, 152, 0, 0.15)',
        
        // Anomaly Detection
        ANOMALY_PRIMARY: '#9c27b0',
        ANOMALY_CONNECTION: '#ce93d8',
        
        // Node Type Colors
        NODE_SOURCE: '#607D8B',        // Blue-gray for sources
        
        // UI Colors (from current theme)
        ACCENT: '#00ccff',
        TEXT_PRIMARY: '#e0e0e0',
        TEXT_SECONDARY: '#a0a0a0',
        BACKGROUND_PRIMARY: '#0a0e17',
        BACKGROUND_SECONDARY: '#141b29'
    },
    
    // Node Shape Configuration
    SHAPES: {
        ENTITY: 'circle',
        CLAIM: 'circle',
        SOURCE: 'square',
        DOCUMENT: 'square',
        UMBRELLA: 'circle-hollow',
        META: 'hexagon'
    },
    
    // Size Configuration
    SIZES: {
        NODE_RADIUS_SMALL: 8,          // Standard nodes (reduced from 12)
        NODE_RADIUS_LARGE: 11,         // High-importance nodes
        NODE_RADIUS_CONTAINER: 50,     // Umbrella containers (variable 40-80)
        
        TEXT_BASE: 11,                 // Base font size (reduced from 14)
        TEXT_HOVER: 13,                // On hover (slightly larger)
        TEXT_CONTAINER: 14,            // Umbrella container labels
        
        EDGE_WIDTH_THIN: 1,
        EDGE_WIDTH_MEDIUM: 1.5,
        EDGE_WIDTH_THICK: 2,
        
        STROKE_WIDTH_NORMAL: 1.5,
        STROKE_WIDTH_SUPPRESSED: 3,
        STROKE_WIDTH_DETECTED: 3
    },
    
    // Edge Style Configuration
    EDGE_STYLES: {
        MENTIONS: {
            color: '#a0a0a0',
            width: 1,
            dasharray: 'none',
            opacity: 0.6
        },
        SUPPORTS: {
            color: '#4CAF50',
            width: 2,
            dasharray: 'none',
            opacity: 0.8
        },
        CONTRADICTS: {
            color: '#E91E63',
            width: 2,
            dasharray: '5,5',
            opacity: 0.8
        },
        CITES: {
            color: '#00ccff',
            width: 1.5,
            dasharray: 'none',
            opacity: 0.7
        },
        CONTAINS: {
            color: '#607D8B',
            width: 1,
            dasharray: 'none',
            opacity: 0.5
        }
    },
    
    // Force Simulation Parameters
    SIMULATION: {
        LINK_DISTANCE: 100,
        CHARGE_STRENGTH: -300,
        COLLISION_RADIUS: 30,
        TICK_THROTTLE: 3,              // Update every 3rd frame
        ALPHA_TARGET: 0,
        ALPHA_DECAY: 0.02,
        VELOCITY_DECAY: 0.4
    },
    
    // Performance Thresholds
    PERFORMANCE: {
        LOD_SPEED_THRESHOLD: 0.5,      // Hide labels above this speed
        FPS_TARGET: 60,
        FPS_ACCEPTABLE: 55,
        NODE_COUNT_WEBGL: 500          // Switch to WebGL above this
    },
    
    // Suppression Detection
    SUPPRESSION: {
        HIGH_THRESHOLD: 0.7,           // Score above which to show indicator
        MEDIUM_THRESHOLD: 0.4,         // Medium suppression threshold
        GLOW_RADIUS: 6,
        BORDER_DASH: '5,5',
        META_OPACITY: 0.4,             // Dim META claims in suppression view
        PRIMARY_BRIGHTNESS: 1.2        // Brighten PRIMARY claims
    },
    
    // Coordination Detection
    COORDINATION: {
        CLUSTER_THRESHOLD: 0.6,
        TEMPORAL_WINDOW_DAYS: 14,
        MIN_CLUSTER_SIZE: 3,
        PULSE_DURATION: 2000,
        CLUSTER_PADDING: 20
    },
    
    // Anomaly Detection
    ANOMALY: {
        DISTANCE_THRESHOLD: 1000,      // km
        CONFIDENCE_THRESHOLD: 0.5,
        CONNECTION_DASHARRAY: '8,4'
    },
    
    // Detection Modes Configuration
    DETECTION_MODES: {
        standard: {
            name: 'Standard',
            icon: '',  // No emoji
            color: '#00ccff',
            description: 'Search the knowledge graph',
            placeholder: 'Search for claims, entities, or topics...',
            buttonText: 'Search',
        },
        suppression: {
            name: 'Detect Suppression',
            icon: '',  // No emoji
            color: '#ff4444',
            description: 'Find high-quality research being systematically ignored',
            placeholder: 'Enter topic to analyze for suppression...',
            buttonText: 'Analyze',
        },
        coordination: {
            name: 'Detect Coordination',
            icon: '',  // No emoji
            color: '#ff9800',
            description: 'Detect synchronized messaging campaigns',
            placeholder: 'Enter topic to detect coordination...',
            buttonText: 'Analyze',
        },
        anomaly: {
            name: 'Detect Anomalies',
            icon: '',  // No emoji
            color: '#9c27b0',
            description: 'Find cross-cultural patterns suggesting lost knowledge',
            placeholder: 'Enter pattern to find anomalies...',
            buttonText: 'Analyze',
        },
    },
    
    // Timeline Configuration
    TIMELINE: {
        SCALES: {
            day: { unit: 'day', format: 'MMM D', ticks: 7 },
            week: { unit: 'week', format: 'MMM D', ticks: 8 },
            month: { unit: 'month', format: 'MMM YYYY', ticks: 12 },
            year: { unit: 'year', format: 'YYYY', ticks: 10 },
            decade: { unit: 'year', format: 'YYYY', ticks: 10, step: 10 },
            century: { unit: 'year', format: 'YYYY', ticks: 10, step: 100 },
        },
        GRID_COLOR: 'rgba(255, 255, 255, 0.1)',
        GRID_MAJOR_COLOR: 'rgba(255, 255, 255, 0.2)',
        AXIS_COLOR: '#4a5568',
        MARGIN: { top: 60, right: 40, bottom: 60, left: 150 },
        ROW_HEIGHT: 40,
        NODE_RADIUS: 6,
        GROUP_BY: {
            source: 'Source/Author',
            claim_type: 'Claim Type',
            topic: 'Topic/Theme',
            domain: 'Domain',
        },
        ANIMATION_SPEED: 500,
    },
    
    // Graph View Defaults
    GRAPH: {
        DEFAULT_TRUST_THRESHOLD: 0.0,
        DEFAULT_MAX_NODES: 50,
        DEFAULT_LAYOUT: 'force',
        MIN_TRUST: 0.0,
        MAX_TRUST: 1.0,
        MIN_NODES: 10,
        MAX_NODES: 200
    },
    
    // Side Panel Configuration
    PANEL: {
        WIDTH_OPEN: 300,
        WIDTH_CLOSED: 0,
        TRANSITION_DURATION: '0.3s',
        DEFAULT_TAB: 'claim'
    },
    
    // Data Wizard Configuration
    WIZARD: {
        SMALL_BATCH_MAX: 50,           // Files cutoff for wizard vs CLI
        PROGRESS_UPDATE_INTERVAL: 1000  // ms
    },
    
    // Export Configuration
    EXPORT: {
        FILENAME_PREFIX: 'aegis_graph',
        FILENAME_DATE_FORMAT: 'YYYY-MM-DD',
        IMAGE_FORMAT: 'png',
        QUALITY: 1.0,
        THEMES: {
            WEB: {
                background: '#0a0e17',
                text: '#e0e0e0',
                metadata: '#a0a0a0'
            },
            PRINT: {
                background: '#ffffff',
                text: '#000000',
                metadata: '#666666'
            }
        }
    },
    
    // Keyboard Shortcuts
    SHORTCUTS: {
        SEARCH: 'Enter',
        CLEAR: 'Escape',
        TOGGLE_PANEL: ' ',             // Space
        ZOOM_IN: '+',
        ZOOM_OUT: '-',
        RESET_VIEW: 'r',
        EXPORT_WEB: 'e',
        EXPORT_PRINT: 'E',             // Shift+E
        TAB_PATTERN: '1',
        TAB_GRAPH: '2',
        TAB_DATA: '3',
        TOGGLE_TIMELINE: 't',          // NEW: Toggle timeline view
        HELP: '?'
    }
};

// Utility function to get trust color
export function getTrustColor(trustScore) {
    if (trustScore === null || trustScore === undefined) {
        return CONFIG.COLORS.TRUST_UNKNOWN;
    }
    if (trustScore >= 0.7) return CONFIG.COLORS.TRUST_HIGH;
    if (trustScore >= 0.5) return CONFIG.COLORS.TRUST_MEDIUM;
    return CONFIG.COLORS.TRUST_LOW;
}

// Utility function to get position color
export function getPositionColor(position) {
    const colors = {
        'supporting': CONFIG.COLORS.POSITION_SUPPORT,
        'support': CONFIG.COLORS.POSITION_SUPPORT,
        'opposing': CONFIG.COLORS.POSITION_OPPOSE,
        'oppose': CONFIG.COLORS.POSITION_OPPOSE,
        'neutral': CONFIG.COLORS.POSITION_NEUTRAL,
        'contextual': CONFIG.COLORS.POSITION_CONTEXT
    };
    return colors[position] || CONFIG.COLORS.POSITION_NEUTRAL;
}

// Utility function to get edge style
export function getEdgeStyle(edgeType) {
    return CONFIG.EDGE_STYLES[edgeType] || CONFIG.EDGE_STYLES.MENTIONS;
}

// Utility function to get score color (for detection results)
export function getScoreColor(score) {
    // Aligned with calibration profile thresholds
    if (score >= 0.75) return '#ff4444';  // Critical - red
    if (score >= 0.55) return '#ff9800';  // High - orange
    if (score >= 0.35) return '#ffc107';  // Moderate - yellow
    return '#4caf50';                      // Low - green
}

// Utility function to get score interpretation
export function getScoreInterpretation(score) {
    // Aligned with calibration profile thresholds
    if (score >= 0.75) return 'CRITICAL';
    if (score >= 0.55) return 'HIGH';
    if (score >= 0.35) return 'MODERATE';
    return 'LOW';
}

// Utility function to get detection mode config
export function getDetectionMode(mode) {
    return CONFIG.DETECTION_MODES[mode] || CONFIG.DETECTION_MODES.standard;
}


// Initialize when admin tab is opened
function initializeAdminTab() {
  if (!adminPanel) {
    adminPanel = new AdminPanel();
  }
}

// Add to your tab switching logic
function switchTab(tabName) {
  // ... existing tab switching code ...

  if (tabName === 'admin') {
    initializeAdminTab();
  }

  // Show/hide tabs
  document.querySelectorAll('.tab-content').forEach(tab => {
    tab.style.display = 'none';
  });
  document.getElementById(`${tabName}-tab`).style.display = 'block';
}

// Wire up tab buttons
document.querySelectorAll('.tab-button').forEach(button => {
  button.addEventListener('click', () => {
    const tab = button.dataset.tab;
    switchTab(tab);
  });
});
