/**
 * Detection Runner (UPDATED v1.1)
 * Handles detection execution, API calls, and state management
 * 
 * CHANGES from v1.0:
 * - Added two-stage search pipeline
 * - Phase 1: Call pattern search to get semantically relevant claims
 * - Phase 2: Pass claim IDs to detection for analysis
 * - Updated progress messages for two-phase process
 * - Better error handling for empty results
 * 
 * Date: November 23, 2025
 */

export class DetectionRunner {
    constructor() {
        this.currentJob = null;
        this.abortController = null;
        this.timeout = 90000; // 90 second timeout (increased for two-stage process)
    }
    
    /**
     * Run detection analysis with semantic search
     * @param {string} mode - Detection mode: 'suppression', 'coordination', 'anomaly', 'standard'
     * @param {string} query - Search query
     * @param {object} options - Additional options
     * @returns {Promise<object>} Detection results
     */
    async runDetection(mode, query, options = {}) {
        // Validate inputs
        if (!mode || !query) {
            throw new Error('Mode and query are required');
        }
        
        // For standard mode, use original pattern search
        if (mode === 'standard') {
            return this.runStandardSearch(query, options);
        }
        
        // Validate detection mode
        if (!['suppression', 'coordination', 'anomaly'].includes(mode)) {
            throw new Error(`Invalid detection mode: ${mode}`);
        }
        
        // Cancel any existing job
        this.cancel();
        
        // Create new abort controller
        this.abortController = new AbortController();
        
        // Create job object
        this.currentJob = {
            mode,
            query,
            startTime: Date.now(),
            status: 'running'
        };
        
        try {
            // =================================================================
            // PHASE 1: Get semantically relevant claims via pattern search
            // =================================================================
            
            this.onProgress?.('searching', 0.1, 'Finding relevant claims with semantic search...');
            
            console.log(`[DetectionRunner] Phase 1: Running pattern search for "${query}"`);
            
            const searchResponse = await fetch('/api/pattern-search', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ 
                    query: query.trim(),
                    max_results: options.maxClaims || 500
                }),
                signal: this.abortController.signal
            });
            
            if (!searchResponse.ok) {
                throw new Error(`Pattern search failed: ${searchResponse.status}`);
            }
            
            const searchResult = await searchResponse.json();
            
            // DEBUG: Log the actual structure
            console.log(`[DetectionRunner] Search result structure:`, {
                hasClaims: !!searchResult.claims,
                claimsLength: searchResult.claims?.length || 0,
                hasNeo4j: !!searchResult.neo4j_results,
                hasEmbedding: !!searchResult.embedding_results,
                hasError: !!searchResult.error,
                totalClaims: searchResult.total_claims,
                keys: Object.keys(searchResult)
            });
            
            // Extract claim IDs from search results
            let claimIds = [];
            
            // Check for error in search result
            if (searchResult.error) {
                console.warn(`[DetectionRunner] Pattern search returned error: ${searchResult.error}`);
            }
            
            // Get IDs from claims array (current API structure)
            if (searchResult.claims && searchResult.claims.length > 0) {
                claimIds = searchResult.claims.map(c => c.id).filter(id => id !== undefined && id !== null);
                console.log(`[DetectionRunner] Found ${claimIds.length} claims from pattern search`);
            }
            // Fallback: Try old structure (neo4j_results + embedding_results)
            else {
                if (searchResult.neo4j_results && searchResult.neo4j_results.length > 0) {
                    const neo4jIds = searchResult.neo4j_results.map(c => c.id);
                    claimIds.push(...neo4jIds);
                    console.log(`[DetectionRunner] Found ${neo4jIds.length} claims from Neo4j results`);
                }
                
                if (searchResult.embedding_results && searchResult.embedding_results.length > 0) {
                    const embeddingIds = searchResult.embedding_results.map(c => c.id);
                    claimIds.push(...embeddingIds);
                    console.log(`[DetectionRunner] Found ${embeddingIds.length} claims from embedding results`);
                }
                
                // Deduplicate
                claimIds = [...new Set(claimIds)];
            }
            
            console.log(`[DetectionRunner] Phase 1 complete: ${claimIds.length} unique claims found`);
            
            if (claimIds.length === 0) {
                throw new Error('No relevant claims found for this query. Try a different search term.');
            }
            
            // =================================================================
            // PHASE 2: Run detection analysis on those claims
            // =================================================================
            
            this.onProgress?.('analyzing', 0.3, `Analyzing ${claimIds.length} claims for ${mode} patterns...`);
            
            console.log(`[DetectionRunner] Phase 2: Running ${mode} detection on ${claimIds.length} claims`);
            
            const endpoint = `/api/detect/${mode}`;
            const body = {
                query: query.trim(),
                claim_ids: claimIds,  // NEW: Pass specific claims to analyze
                options: {
                    max_claims: options.maxClaims || 500,
                    include_graph: options.includeGraph !== false,
                    ...options
                }
            };
            
            const timeoutPromise = new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Detection timeout')), this.timeout)
            );
            
            const fetchPromise = fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(body),
                signal: this.abortController.signal
            });
            
            this.onProgress?.('calculating', 0.6, 'Calculating detection signals...');
            
            const response = await Promise.race([fetchPromise, timeoutPromise]);
            
            if (!response.ok) {
                const error = await response.json().catch(() => ({ error: 'Unknown error' }));
                throw new Error(error.error || `HTTP ${response.status}`);
            }
            
            this.onProgress?.('formatting', 0.9, 'Generating interpretation...');
            
            const result = await response.json();
            
            if (!result.success) {
                throw new Error(result.error || 'Detection failed');
            }
            
            // Add metadata
            result._meta = {
                mode,
                query,
                duration: Date.now() - this.currentJob.startTime,
                timestamp: new Date().toISOString(),
                claims_searched: claimIds.length,
                two_stage_pipeline: true
            };
            
            this.currentJob.status = 'completed';
            this.currentJob.result = result;
            
            console.log(`[DetectionRunner] Detection complete: Score=${result[mode + '_score'] || 'N/A'}, Duration=${result._meta.duration}ms`);
            
            this.onProgress?.('complete', 1.0, 'Complete');
            this.onComplete?.(result);
            
            return result;
            
        } catch (error) {
            this.currentJob.status = 'failed';
            this.currentJob.error = error.message;
            
            console.error(`[DetectionRunner] Error:`, error);
            
            // Handle specific error types
            if (error.name === 'AbortError') {
                this.onError?.(new Error('Detection cancelled'));
            } else if (error.message === 'Detection timeout') {
                this.onError?.(new Error('Detection timed out after 90 seconds'));
            } else if (error.message.includes('fetch') || error.message.includes('Failed to fetch')) {
                this.onError?.(new Error('Network error - is the server running?'));
            } else {
                this.onError?.(error);
            }
            
            throw error;
            
        } finally {
            this.abortController = null;
        }
    }
    
    /**
     * Run standard pattern search (no detection)
     * @param {string} query - Search query
     * @param {object} options - Additional options
     * @returns {Promise<object>} Search results
     */
    async runStandardSearch(query, options = {}) {
        console.log(`[DetectionRunner] Running standard search for "${query}"`);
        
        this.currentJob = {
            mode: 'standard',
            query,
            startTime: Date.now(),
            status: 'running'
        };
        
        try {
            this.onProgress?.('searching', 0.3, 'Searching knowledge graph...');
            
            const response = await fetch('/api/pattern-search', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ 
                    query: query.trim(),
                    max_results: options.maxClaims || 500
                })
            });
            
            if (!response.ok) {
                throw new Error(`Search failed: ${response.status}`);
            }
            
            this.onProgress?.('analyzing', 0.7, 'Processing results...');
            
            const result = await response.json();
            
            // Format result for consistency with detection modes
            const formattedResult = {
                success: true,
                mode: 'standard',
                query: query,
                neo4j_results: result.neo4j_results || [],
                embedding_results: result.embedding_results || [],
                llm_synthesis: result.llm_synthesis || null,
                total_claims: (result.neo4j_results?.length || 0) + (result.embedding_results?.length || 0),
                _meta: {
                    mode: 'standard',
                    query,
                    duration: Date.now() - this.currentJob.startTime,
                    timestamp: new Date().toISOString()
                }
            };
            
            this.currentJob.status = 'completed';
            this.currentJob.result = formattedResult;
            
            this.onProgress?.('complete', 1.0, 'Complete');
            this.onComplete?.(formattedResult);
            
            return formattedResult;
            
        } catch (error) {
            this.currentJob.status = 'failed';
            this.currentJob.error = error.message;
            
            this.onError?.(error);
            throw error;
        }
    }
    
    /**
     * Cancel current detection job
     */
    cancel() {
        if (this.abortController) {
            this.abortController.abort();
            this.abortController = null;
        }
        
        if (this.currentJob && this.currentJob.status === 'running') {
            this.currentJob.status = 'cancelled';
            this.onCancelled?.();
        }
    }
    
    /**
     * Check if detection is currently running
     */
    isRunning() {
        return this.currentJob?.status === 'running';
    }
    
    /**
     * Get current job status
     */
    getStatus() {
        return this.currentJob;
    }
    
    /**
     * Set progress callback
     * @param {function} callback - Called with (phase, progress, message)
     */
    setProgressCallback(callback) {
        this.onProgress = callback;
    }
    
    /**
     * Set complete callback
     * @param {function} callback - Called with (result)
     */
    setCompleteCallback(callback) {
        this.onComplete = callback;
    }
    
    /**
     * Set error callback
     * @param {function} callback - Called with (error)
     */
    setErrorCallback(callback) {
        this.onError = callback;
    }
    
    /**
     * Set cancelled callback
     * @param {function} callback - Called when detection is cancelled
     */
    setCancelledCallback(callback) {
        this.onCancelled = callback;
    }
}

/**
 * Detection Loading State Component
 * Shows progress during detection
 */
export class DetectionLoadingState {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.currentPhase = null;
        this.currentProgress = 0;
    }
    
    show(phase, progress, message) {
        if (!this.container) return;
        
        this.currentPhase = phase;
        this.currentProgress = progress;
        
        const progressPercent = Math.round(progress * 100);
        
        const html = `
            <div class="detection-loading">
                <div class="loading-icon">
                    <div class="spinner"></div>
                </div>
                <h3 class="loading-title">Analyzing...</h3>
                <div class="progress-bar-container">
                    <div class="progress-bar" style="width: ${progressPercent}%"></div>
                    <div class="progress-text">${progressPercent}%</div>
                </div>
                <p class="loading-message">${this.escapeHtml(message)}</p>
                <div class="loading-stats">
                    <span class="stat">Phase: ${this.getPhaseLabel(phase)}</span>
                    <span class="stat">Estimated: ${this.getEstimatedTime(progress)}</span>
                </div>
                <button class="btn-cancel" id="cancel-detection">Cancel</button>
            </div>
        `;
        
        this.container.innerHTML = html;
        this.container.style.display = 'block';
        
        // Attach cancel handler
        document.getElementById('cancel-detection')?.addEventListener('click', () => {
            this.onCancel?.();
        });
    }
    
    hide() {
        if (this.container) {
            this.container.style.display = 'none';
            this.container.innerHTML = '';
        }
    }
    
    getPhaseLabel(phase) {
        const labels = {
            'searching': 'Searching',
            'analyzing': 'Analyzing',
            'calculating': 'Calculating',
            'formatting': 'Formatting',
            'complete': 'Complete'
        };
        return labels[phase] || 'Processing';
    }
    
    getEstimatedTime(progress) {
        if (progress >= 0.9) return '< 5 seconds';
        if (progress >= 0.6) return '10-15 seconds';
        if (progress >= 0.3) return '15-25 seconds';
        return '25-40 seconds';
    }
    
    setCancelCallback(callback) {
        this.onCancel = callback;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}
