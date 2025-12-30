/**
 * Aegis Insight v2.0 - Search Synchronization
 * Synchronizes search queries across Pattern Search and Graph Search tabs
 */

export class SearchSync {
    constructor() {
        this.currentQuery = '';
        this.listeners = new Map();
    }
    
    /**
     * Register a search input element
     * @param {string} id - Unique identifier for this input
     * @param {HTMLElement} inputElement - The input element
     * @param {HTMLElement} searchButton - The search button to trigger
     */
    register(id, inputElement, searchButton) {
        if (!inputElement) {
            console.warn(`SearchSync: Input element not found for ${id}`);
            return;
        }
        
        // Store reference
        this.listeners.set(id, { inputElement, searchButton });
        
        // Listen to input changes
        inputElement.addEventListener('input', (e) => {
            this.handleInputChange(e.target.value, id);
        });
        
        // Handle Enter key
        inputElement.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                if (searchButton) {
                    searchButton.click();
                }
            }
        });
        
        // Initialize with current query if exists
        if (this.currentQuery) {
            inputElement.value = this.currentQuery;
        }
    }
    
    /**
     * Handle input change from any source
     * @param {string} newQuery - New query value
     * @param {string} sourceId - ID of the source that changed
     */
    handleInputChange(newQuery, sourceId) {
        if (this.currentQuery === newQuery) return; // No change
        
        this.currentQuery = newQuery;
        
        // Update all other registered inputs (not the source)
        this.listeners.forEach((listener, id) => {
            if (id !== sourceId) {
                listener.inputElement.value = newQuery;
            }
        });
    }
    
    /**
     * Programmatically set query (updates all inputs)
     * @param {string} query - Query to set
     */
    setQuery(query) {
        this.currentQuery = query;
        
        // Update all registered inputs
        this.listeners.forEach(listener => {
            listener.inputElement.value = query;
        });
    }
    
    /**
     * Get current query
     * @returns {string} Current query
     */
    getQuery() {
        return this.currentQuery;
    }
    
    /**
     * Clear query from all inputs
     */
    clear() {
        this.setQuery('');
    }
    
    /**
     * Trigger search for a specific tab
     * @param {string} id - ID of the search to trigger
     */
    triggerSearch(id) {
        const listener = this.listeners.get(id);
        if (listener && listener.searchButton) {
            listener.searchButton.click();
        }
    }
}
