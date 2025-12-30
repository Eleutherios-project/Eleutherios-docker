// admin/AdminPanel.js
// Admin UI Controller for Detection System Setup

export class AdminPanel {
  constructor() {
    this.statusPollInterval = null;
    this.jobProgressPoll = null;
    this.activeJobs = new Map();
    this.initialize();
  }

  initialize() {
    this.setupEventListeners();
    this.refreshStatus();
    // Poll status every 30 seconds
    this.statusPollInterval = setInterval(() => this.refreshStatus(), 30000);
  }

  setupEventListeners() {
    // Refresh status button
    document.getElementById('refresh-status')?.addEventListener('click', () => {
      this.refreshStatus();
      this.showNotification('Status refreshed', 'info');
    });

    // Setup all detectors
    document.getElementById('setup-all-detectors')?.addEventListener('click', () => {
      this.setupAllDetectors();
    });

    // Individual detector setup buttons
    document.querySelectorAll('.setup-detector').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const detector = e.target.dataset.detector;
        this.setupDetector(detector);
      });
    });

    // Clear checkpoints
    document.getElementById('clear-checkpoints')?.addEventListener('click', () => {
      this.clearCheckpoints();
    });

    // View logs
    document.getElementById('view-logs')?.addEventListener('click', () => {
      this.viewLogs();
    });

    // Minimize jobs panel
    document.getElementById('minimize-jobs')?.addEventListener('click', () => {
      this.toggleJobsPanel();
    });
  }

  async refreshStatus() {
    try {
      const response = await fetch('/api/system/status');
      const status = await response.json();
      
      this.updateSystemStats(status);
      this.updateDetectorStatus(status);
      this.updateSystemInfo(status);
      
    } catch (error) {
      console.error('Failed to fetch system status:', error);
      this.showNotification('Failed to refresh status', 'error');
    }
  }

  updateSystemStats(status) {
    // Update stat cards
    document.getElementById('total-claims').textContent = 
      status.claims_count?.toLocaleString() || '---';
    
    document.getElementById('total-citations').textContent = 
      status.citations_count?.toLocaleString() || '---';
    
    document.getElementById('total-embeddings').textContent = 
      status.embeddings_count?.toLocaleString() || '---';
    
    document.getElementById('geo-claims').textContent = 
      status.geographic_claims_count?.toLocaleString() || '---';
  }

  updateDetectorStatus(status) {
    const detectors = ['suppression', 'coordination', 'anomaly'];
    
    detectors.forEach(detector => {
      const detectorStatus = status.detectors_ready?.[detector];
      if (!detectorStatus) return;

      // Update status badge
      const statusBadge = document.getElementById(`${detector}-status`);
      const requirementsDiv = document.getElementById(`${detector}-requirements`);
      const actionsDiv = document.getElementById(`${detector}-actions`);
      const card = document.getElementById(`${detector}-detector`);

      if (detectorStatus.ready) {
        // Detector is ready
        statusBadge.textContent = '✓ Ready';
        statusBadge.className = 'status-badge ready';
        card.classList.remove('needs-setup');
        card.classList.add('ready');
        
        requirementsDiv.innerHTML = `
          <div class="requirement-item">
            <span class="requirement-status">✓</span>
            <span class="requirement-text">All requirements met</span>
          </div>
        `;
        
        actionsDiv.style.display = 'none';
      } else {
        // Detector needs setup
        statusBadge.textContent = '⚠️ Needs Setup';
        statusBadge.className = 'status-badge needs-setup';
        card.classList.remove('ready');
        card.classList.add('needs-setup');
        
        // Show missing requirements
        const missing = detectorStatus.missing || [];
        requirementsDiv.innerHTML = missing.map(job => {
          const jobInfo = this.getJobInfo(job);
          return `
            <div class="requirement-item">
              <span class="requirement-status">⏳</span>
              <span class="requirement-text">${jobInfo.name}</span>
              <span class="requirement-time">~${jobInfo.time}</span>
            </div>
          `;
        }).join('');
        
        actionsDiv.style.display = 'flex';
      }
    });
  }

  updateSystemInfo(status) {
    // Neo4j status
    const neo4jStatus = document.getElementById('neo4j-status');
    if (status.neo4j_connected) {
      neo4jStatus.textContent = '✓ Connected';
      neo4jStatus.className = 'info-value online';
    } else {
      neo4jStatus.textContent = '✗ Disconnected';
      neo4jStatus.className = 'info-value offline';
    }

    // Database size
    document.getElementById('db-size').textContent = 
      status.db_size || '---';

    // Last ingestion
    if (status.last_ingestion) {
      const date = new Date(status.last_ingestion);
      document.getElementById('last-ingestion').textContent = 
        date.toLocaleString();
    } else {
      document.getElementById('last-ingestion').textContent = 'Never';
    }

    // Ollama status
    const ollamaStatus = document.getElementById('ollama-status');
    if (status.ollama_available) {
      ollamaStatus.textContent = '✓ Available';
      ollamaStatus.className = 'info-value online';
    } else {
      ollamaStatus.textContent = '✗ Unavailable';
      ollamaStatus.className = 'info-value offline';
    }
  }

  async setupDetector(detector) {
    this.showNotification(`Starting setup for ${detector} detector...`, 'info');

    try {
      const response = await fetch('/api/admin/detection-setup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ detector })
      });

      const result = await response.json();

      if (result.success) {
        this.showNotification(
          `${detector} detector setup started`,
          'success'
        );
        
        // Show job progress panel
        this.showJobProgressPanel();
        
        // Start polling for job progress
        this.startJobProgressPolling(result.job_ids);
      } else {
        this.showNotification(
          `Failed to start ${detector} setup: ${result.error}`,
          'error'
        );
      }
    } catch (error) {
      console.error('Setup failed:', error);
      this.showNotification(`Setup failed: ${error.message}`, 'error');
    }
  }

  async setupAllDetectors() {
    const confirmed = confirm(
      'This will run setup for all three detectors. ' +
      'This may take 30-60 minutes. Continue?'
    );

    if (!confirmed) return;

    this.showNotification('Starting setup for all detectors...', 'info');

    try {
      const response = await fetch('/api/admin/detection-setup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ all: true })
      });

      const result = await response.json();

      if (result.success) {
        this.showNotification(
          'All detector setups started. This will take ~30-60 minutes.',
          'success'
        );
        
        this.showJobProgressPanel();
        this.startJobProgressPolling(result.job_ids);
      } else {
        this.showNotification(
          `Failed to start setup: ${result.error}`,
          'error'
        );
      }
    } catch (error) {
      console.error('Setup failed:', error);
      this.showNotification(`Setup failed: ${error.message}`, 'error');
    }
  }

  showJobProgressPanel() {
    const panel = document.getElementById('job-progress-panel');
    panel.style.display = 'flex';
  }

  toggleJobsPanel() {
    const panel = document.getElementById('job-progress-panel');
    panel.classList.toggle('minimized');
    
    const btn = document.getElementById('minimize-jobs');
    btn.textContent = panel.classList.contains('minimized') ? '▲' : '▼';
  }

  startJobProgressPolling(jobIds) {
    // Clear any existing poll
    if (this.jobProgressPoll) {
      clearInterval(this.jobProgressPoll);
    }

    // Initialize jobs
    jobIds.forEach(jobId => {
      this.activeJobs.set(jobId, { status: 'running', progress: 0 });
      this.addJobToUI(jobId);
    });

    // Poll every 2 seconds
    this.jobProgressPoll = setInterval(() => {
      this.updateJobProgress();
    }, 2000);
  }

  async updateJobProgress() {
    try {
      const response = await fetch('/api/system/status');
      const status = await response.json();

      let allComplete = true;

      this.activeJobs.forEach((jobInfo, jobId) => {
        const jobStatus = status.jobs?.[jobId];
        
        if (!jobStatus) return;

        if (jobStatus.status === 'running') {
          allComplete = false;
        }

        // Update job info
        this.activeJobs.set(jobId, jobStatus);
        this.updateJobInUI(jobId, jobStatus);
      });

      // If all jobs complete, stop polling
      if (allComplete) {
        clearInterval(this.jobProgressPoll);
        this.jobProgressPoll = null;
        
        this.showNotification('All jobs completed!', 'success');
        
        // Refresh status after jobs complete
        setTimeout(() => this.refreshStatus(), 2000);
      }
    } catch (error) {
      console.error('Failed to update job progress:', error);
    }
  }

  addJobToUI(jobId) {
    const container = document.getElementById('jobs-container');
    
    const jobElement = document.createElement('div');
    jobElement.className = 'job-item';
    jobElement.id = `job-${jobId}`;
    jobElement.innerHTML = `
      <div class="job-header">
        <span class="job-name">${this.getJobInfo(jobId).name}</span>
        <span class="job-status running">Running</span>
      </div>
      <div class="progress-bar-container">
        <div class="progress-bar" style="width: 0%"></div>
      </div>
      <div class="job-details">
        <span class="job-progress-text">0%</span>
        <span class="job-eta">Calculating...</span>
      </div>
    `;
    
    container.appendChild(jobElement);
  }

  updateJobInUI(jobId, jobStatus) {
    const jobElement = document.getElementById(`job-${jobId}`);
    if (!jobElement) return;

    const progress = Math.round((jobStatus.progress || 0) * 100);
    
    // Update status badge
    const statusBadge = jobElement.querySelector('.job-status');
    statusBadge.textContent = jobStatus.status;
    statusBadge.className = `job-status ${jobStatus.status}`;

    // Update progress bar
    const progressBar = jobElement.querySelector('.progress-bar');
    progressBar.style.width = `${progress}%`;

    // Update progress text
    const progressText = jobElement.querySelector('.job-progress-text');
    progressText.textContent = `${progress}%`;

    // Update ETA
    const etaElement = jobElement.querySelector('.job-eta');
    if (jobStatus.eta) {
      etaElement.textContent = `ETA: ${jobStatus.eta}`;
    } else if (jobStatus.status === 'complete') {
      etaElement.textContent = 'Complete';
    } else if (jobStatus.status === 'failed') {
      etaElement.textContent = 'Failed';
    }
  }

  getJobInfo(jobId) {
    const jobMap = {
      'build_citations': {
        name: 'Build Citation Network',
        time: '15-40 min'
      },
      'generate_embeddings': {
        name: 'Generate Embeddings',
        time: '10-15 min'
      },
      'build_temporal_index': {
        name: 'Build Temporal Index',
        time: '5-10 min'
      },
      'build_geographic_index': {
        name: 'Build Geographic Index',
        time: '5-10 min'
      }
    };

    return jobMap[jobId] || { name: jobId, time: 'Unknown' };
  }

  showNotification(message, type = 'info') {
    const container = document.getElementById('action-notifications');
    
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
      <div class="notification-header">
        <span class="notification-title">${this.getNotificationTitle(type)}</span>
        <button class="notification-close">×</button>
      </div>
      <div class="notification-body">${message}</div>
    `;
    
    container.appendChild(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
      notification.style.animation = 'slideOut 0.3s ease';
      setTimeout(() => notification.remove(), 300);
    }, 5000);

    // Close button
    notification.querySelector('.notification-close').addEventListener('click', () => {
      notification.remove();
    });
  }

  getNotificationTitle(type) {
    const titles = {
      'success': '✓ Success',
      'error': '✗ Error',
      'warning': '⚠️ Warning',
      'info': 'ℹ️ Info'
    };
    return titles[type] || 'Notification';
  }

  async clearCheckpoints() {
    const confirmed = confirm(
      'This will clear all job checkpoints. ' +
      'Jobs will restart from the beginning. Continue?'
    );

    if (!confirmed) return;

    try {
      const response = await fetch('/api/admin/clear-checkpoints', {
        method: 'POST'
      });

      if (response.ok) {
        this.showNotification('Checkpoints cleared', 'success');
      } else {
        this.showNotification('Failed to clear checkpoints', 'error');
      }
    } catch (error) {
      console.error('Failed to clear checkpoints:', error);
      this.showNotification(`Failed: ${error.message}`, 'error');
    }
  }

  viewLogs() {
    // Open logs in new window/tab
    window.open('/api/admin/logs', '_blank');
  }

  destroy() {
    if (this.statusPollInterval) {
      clearInterval(this.statusPollInterval);
    }
    if (this.jobProgressPoll) {
      clearInterval(this.jobProgressPoll);
    }
  }
}

// CSS animation for slide out
const style = document.createElement('style');
style.textContent = `
  @keyframes slideOut {
    to {
      transform: translateX(400px);
      opacity: 0;
    }
  }
`;
document.head.appendChild(style);
