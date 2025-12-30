#!/usr/bin/env python3
"""
AegisTrustNet Data Loading Configuration UI

Simple web interface for configuring and running the data pipeline.
No complex dependencies - just Python standard library + what you already have.

Usage:
    python3 aegis_config_ui.py
    
Then open: http://localhost:8083
"""

import http.server
import socketserver
import json
import urllib.parse
import subprocess
import sys
from pathlib import Path
import threading

PORT = 8083

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AegisTrustNet Data Loader</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }
        
        h1 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        .section {
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .section h2 {
            color: #764ba2;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 600;
        }
        
        input[type="text"],
        input[type="password"],
        select {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        
        input[type="text"]:focus,
        input[type="password"]:focus,
        select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .checkbox-group {
            margin: 15px 0;
        }
        
        .checkbox-item {
            margin: 10px 0;
        }
        
        input[type="checkbox"] {
            width: 20px;
            height: 20px;
            margin-right: 10px;
            cursor: pointer;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1em;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            width: 100%;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        .status {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            display: none;
        }
        
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 2px solid #c3e6cb;
        }
        
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 2px solid #f5c6cb;
        }
        
        .status.info {
            background: #d1ecf1;
            color: #0c5460;
            border: 2px solid #bee5eb;
        }
        
        .help-text {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
        
        .tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 2px solid #ddd;
        }
        
        .tab {
            padding: 15px 30px;
            cursor: pointer;
            border: none;
            background: none;
            color: #666;
            font-weight: 600;
            transition: color 0.3s;
            width: auto;
        }
        
        .tab.active {
            color: #667eea;
            border-bottom: 3px solid #667eea;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ AegisTrustNet</h1>
        <p class="subtitle">Data Loading Configuration</p>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('simple')">Simple</button>
            <button class="tab" onclick="showTab('advanced')">Advanced</button>
        </div>
        
        <!-- Simple Tab -->
        <div id="simple" class="tab-content active">
            <form id="simpleForm" onsubmit="loadDataSimple(event)">
                <div class="section">
                    <h2>📂 Data Source</h2>
                    
                    <label>
                        <input type="radio" name="source_type" value="pdfs" checked onchange="toggleSource()">
                        I have PDF files
                    </label>
                    
                    <label>
                        <input type="radio" name="source_type" value="jsonl" onchange="toggleSource()">
                        I already have JSONL files
                    </label>
                    
                    <div id="pdf_input" style="margin-top: 15px;">
                        <label>PDF Directory:</label>
                        <input type="text" id="pdf_dir" placeholder="/path/to/your/pdfs">
                        <p class="help-text">Full path to folder containing your PDF files</p>
                    </div>
                    
                    <div id="jsonl_input" style="margin-top: 15px; display: none;">
                        <label>JSONL Directory:</label>
                        <input type="text" id="jsonl_dir" placeholder="/path/to/your/jsonl">
                        <p class="help-text">Full path to folder containing JSONL files</p>
                    </div>
                </div>
                
                <button type="submit">🚀 Load Data into AegisTrustNet</button>
            </form>
        </div>
        
        <!-- Advanced Tab -->
        <div id="advanced" class="tab-content">
            <form id="advancedForm" onsubmit="loadDataAdvanced(event)">
                <div class="section">
                    <h2>📂 Data Source</h2>
                    
                    <label>PDF Directory (optional):</label>
                    <input type="text" id="adv_pdf_dir" placeholder="/path/to/pdfs">
                    
                    <label>JSONL Directory (optional):</label>
                    <input type="text" id="adv_jsonl_dir" placeholder="/path/to/jsonl">
                    
                    <label>Training Script (optional):</label>
                    <input type="text" id="training_script" placeholder="/path/to/training_pipeline.py">
                    <p class="help-text">Your custom PDF → JSONL processor</p>
                    
                    <label>Output Directory:</label>
                    <input type="text" id="output_dir" value="./aegis_processed">
                </div>
                
                <div class="section">
                    <h2>⚙️ Processing Options</h2>
                    
                    <div class="checkbox-item">
                        <input type="checkbox" id="skip_generation">
                        <label for="skip_generation" style="display: inline;">Skip JSONL generation</label>
                        <p class="help-text">Check if you already have JSONL files</p>
                    </div>
                    
                    <div class="checkbox-item">
                        <input type="checkbox" id="skip_loading">
                        <label for="skip_loading" style="display: inline;">Skip Neo4j loading</label>
                        <p class="help-text">Only generate JSONL, don't load to database</p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔧 Neo4j Connection</h2>
                    
                    <label>Neo4j URI:</label>
                    <input type="text" id="neo4j_uri" value="bolt://localhost:7687">
                    
                    <label>Username:</label>
                    <input type="text" id="neo4j_user" value="neo4j">
                    
                    <label>Password:</label>
                    <input type="password" id="neo4j_password" value="aegistrusted">
                </div>
                
                <button type="submit">🚀 Run Pipeline</button>
            </form>
        </div>
        
        <div id="status" class="status"></div>
    </div>
    
    <script>
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }
        
        function toggleSource() {
            const sourceType = document.querySelector('input[name="source_type"]:checked').value;
            document.getElementById('pdf_input').style.display = sourceType === 'pdfs' ? 'block' : 'none';
            document.getElementById('jsonl_input').style.display = sourceType === 'jsonl' ? 'block' : 'none';
        }
        
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + type;
            status.style.display = 'block';
        }
        
        async function loadDataSimple(event) {
            event.preventDefault();
            
            const sourceType = document.querySelector('input[name="source_type"]:checked').value;
            const path = sourceType === 'pdfs' 
                ? document.getElementById('pdf_dir').value
                : document.getElementById('jsonl_dir').value;
            
            if (!path) {
                showStatus('Please enter a directory path', 'error');
                return;
            }
            
            showStatus('Starting pipeline... This may take a while.', 'info');
            
            const params = {
                type: sourceType,
                path: path
            };
            
            try {
                const response = await fetch('/run_pipeline', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(params)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showStatus('✓ Pipeline completed successfully! Open http://localhost:8082 to explore your data.', 'success');
                } else {
                    showStatus('✗ Pipeline failed: ' + result.error, 'error');
                }
            } catch (error) {
                showStatus('✗ Error: ' + error.message, 'error');
            }
        }
        
        async function loadDataAdvanced(event) {
            event.preventDefault();
            
            showStatus('Starting pipeline... This may take a while.', 'info');
            
            const params = {
                pdf_dir: document.getElementById('adv_pdf_dir').value,
                jsonl_dir: document.getElementById('adv_jsonl_dir').value,
                training_script: document.getElementById('training_script').value,
                output_dir: document.getElementById('output_dir').value,
                skip_generation: document.getElementById('skip_generation').checked,
                skip_loading: document.getElementById('skip_loading').checked,
                neo4j_uri: document.getElementById('neo4j_uri').value,
                neo4j_user: document.getElementById('neo4j_user').value,
                neo4j_password: document.getElementById('neo4j_password').value
            };
            
            try {
                const response = await fetch('/run_pipeline', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(params)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showStatus('✓ Pipeline completed successfully!', 'success');
                } else {
                    showStatus('✗ Pipeline failed: ' + result.error, 'error');
                }
            } catch (error) {
                showStatus('✗ Error: ' + error.message, 'error');
            }
        }
    </script>
</body>
</html>
"""


class PipelineHandler(http.server.SimpleHTTPRequestHandler):
    """Handle pipeline configuration requests"""
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/run_pipeline':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode())
            
            # Build command
            cmd = ['python3', 'aegis_load_pipeline.py']
            
            if 'type' in params:
                # Simple mode
                if params['type'] == 'pdfs':
                    cmd.extend(['--pdfs', params['path']])
                else:
                    cmd.extend(['--jsonl', params['path']])
            else:
                # Advanced mode
                if params.get('pdf_dir'):
                    cmd.extend(['--pdfs', params['pdf_dir']])
                elif params.get('jsonl_dir'):
                    cmd.extend(['--jsonl', params['jsonl_dir']])
                
                if params.get('training_script'):
                    cmd.extend(['--training-script', params['training_script']])
                
                if params.get('output_dir'):
                    cmd.extend(['--output-dir', params['output_dir']])
                
                if params.get('skip_generation'):
                    cmd.append('--skip-generation')
                
                if params.get('skip_loading'):
                    cmd.append('--skip-loading')
                
                if params.get('neo4j_uri'):
                    cmd.extend(['--neo4j-uri', params['neo4j_uri']])
                
                if params.get('neo4j_user'):
                    cmd.extend(['--neo4j-user', params['neo4j_user']])
                
                if params.get('neo4j_password'):
                    cmd.extend(['--neo4j-password', params['neo4j_password']])
            
            # Run pipeline in background
            def run_pipeline():
                try:
                    subprocess.run(cmd, check=True)
                except Exception as e:
                    print(f"Pipeline error: {e}")
            
            thread = threading.Thread(target=run_pipeline)
            thread.start()
            
            # Send response
            response = {'success': True, 'message': 'Pipeline started'}
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass


def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "AEGISTRUSTNET CONFIGURATION UI" + " " * 22 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"🌐 Starting web interface on http://localhost:{PORT}")
    print()
    print("Open your browser to configure and run the data pipeline.")
    print("Press Ctrl+C to stop.")
    print()
    
    with socketserver.TCPServer(("", PORT), PipelineHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            sys.exit(0)


if __name__ == '__main__':
    main()
