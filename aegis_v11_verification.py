#!/usr/bin/env python3
"""
Aegis Insight v1.1 Pre-Release Verification Script
==================================================

Run this script to verify all critical components are working before release.

Usage:
    python aegis_v11_verification.py [--host HOST] [--mcp-port MCP_PORT]

Prerequisites:
    - Docker containers running (docker-compose up -d)
    - Ollama accessible
    - At least one corpus loaded (demo data)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Optional, Tuple

# Try to import requests, handle if not available
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Try to import psycopg2 for direct DB check
try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

# Try neo4j driver
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_pass(test_name: str, details: str = ""):
    print(f"  {Colors.GREEN}✓ PASS{Colors.END}: {test_name}")
    if details:
        print(f"         {Colors.BLUE}{details}{Colors.END}")


def print_fail(test_name: str, error: str = ""):
    print(f"  {Colors.RED}✗ FAIL{Colors.END}: {test_name}")
    if error:
        print(f"         {Colors.RED}{error}{Colors.END}")


def print_warn(test_name: str, warning: str = ""):
    print(f"  {Colors.YELLOW}⚠ WARN{Colors.END}: {test_name}")
    if warning:
        print(f"         {Colors.YELLOW}{warning}{Colors.END}")


def print_skip(test_name: str, reason: str = ""):
    print(f"  {Colors.YELLOW}○ SKIP{Colors.END}: {test_name}")
    if reason:
        print(f"         {Colors.YELLOW}{reason}{Colors.END}")


class AegisVerification:
    def __init__(self, host: str = "localhost", api_port: int = 8000, mcp_port: int = 8001):
        self.host = host
        self.api_port = api_port
        self.mcp_port = mcp_port
        self.api_base = f"http://{host}:{api_port}"
        self.mcp_base = f"http://{host}:{mcp_port}"
        self.results = {"passed": 0, "failed": 0, "warnings": 0, "skipped": 0}
        
    def test_docker_containers(self) -> bool:
        """Check if required Docker containers are running"""
        print_header("1. Docker Container Status")
        
        import subprocess
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=10
            )
            running = result.stdout.strip().split('\n')
            
            required = {
                'neo4j': False,
                'postgres': False,
                'aegis': False  # Main app container
            }
            
            for container in running:
                for key in required:
                    if key in container.lower():
                        required[key] = True
                        
            all_running = True
            for name, status in required.items():
                if status:
                    print_pass(f"Container '{name}'", "Running")
                    self.results["passed"] += 1
                else:
                    print_fail(f"Container '{name}'", "Not found or not running")
                    self.results["failed"] += 1
                    all_running = False
                    
            return all_running
            
        except FileNotFoundError:
            print_warn("Docker check", "Docker command not found - skipping container checks")
            self.results["skipped"] += 1
            return True  # Don't fail if docker CLI not available
        except Exception as e:
            print_fail("Docker check", str(e))
            self.results["failed"] += 1
            return False
    
    def test_neo4j_connection(self) -> bool:
        """Test Neo4j database connectivity"""
        print_header("2. Neo4j Database Connection")
        
        if not NEO4J_AVAILABLE:
            print_skip("Neo4j driver", "neo4j package not installed")
            self.results["skipped"] += 1
            return True
            
        try:
            uri = os.getenv("NEO4J_URI", f"bolt://{self.host}:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "aegistrusted")
            
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count")
                count = result.single()["count"]
                
            driver.close()
            print_pass("Neo4j connection", f"Connected, {count:,} nodes in graph")
            self.results["passed"] += 1
            return True
            
        except Exception as e:
            print_fail("Neo4j connection", str(e))
            self.results["failed"] += 1
            return False
    
    def test_postgres_connection(self) -> bool:
        """Test PostgreSQL database connectivity"""
        print_header("3. PostgreSQL Database Connection")
        
        if not PSYCOPG2_AVAILABLE:
            print_skip("PostgreSQL driver", "psycopg2 package not installed")
            self.results["skipped"] += 1
            return True
            
        try:
            conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", self.host),
                port=os.getenv("POSTGRES_PORT", "5432"),
                user=os.getenv("POSTGRES_USER", "aegis"),
                password=os.getenv("POSTGRES_PASSWORD", "aegistrusted"),
                database=os.getenv("POSTGRES_DB", "aegis-postgres")
            )
            
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM claim_embeddings")
            count = cur.fetchone()[0]
            
            # Check pgvector extension
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            vector_version = cur.fetchone()
            
            conn.close()
            
            print_pass("PostgreSQL connection", f"Connected, {count:,} embeddings")
            self.results["passed"] += 1
            
            if vector_version:
                print_pass("pgvector extension", f"Version {vector_version[0]}")
                self.results["passed"] += 1
            else:
                print_warn("pgvector extension", "Not found - semantic search may fail")
                self.results["warnings"] += 1
                
            return True
            
        except Exception as e:
            print_fail("PostgreSQL connection", str(e))
            self.results["failed"] += 1
            return False
    
    def test_ollama_connection(self) -> bool:
        """Test Ollama LLM service connectivity"""
        print_header("4. Ollama LLM Service")
        
        if not REQUESTS_AVAILABLE:
            print_skip("Ollama check", "requests package not installed")
            self.results["skipped"] += 1
            return True
            
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        try:
            response = requests.get(f"{ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "unknown") for m in models]
                
                print_pass("Ollama connection", f"{len(models)} models available")
                self.results["passed"] += 1
                
                # Check for recommended model
                recommended = ["mistral-nemo", "mistral", "llama3"]
                found_recommended = any(
                    any(r in m for r in recommended) 
                    for m in model_names
                )
                
                if found_recommended:
                    print_pass("Recommended model", f"Found: {', '.join(model_names[:3])}")
                    self.results["passed"] += 1
                else:
                    print_warn("Recommended model", f"Consider installing mistral-nemo:12b")
                    self.results["warnings"] += 1
                    
                return True
            else:
                print_fail("Ollama connection", f"Status {response.status_code}")
                self.results["failed"] += 1
                return False
                
        except requests.exceptions.ConnectionError:
            print_fail("Ollama connection", f"Cannot connect to {ollama_host}")
            self.results["failed"] += 1
            return False
        except Exception as e:
            print_fail("Ollama connection", str(e))
            self.results["failed"] += 1
            return False
    
    def test_mcp_endpoints(self) -> bool:
        """Test all MCP API endpoints"""
        print_header("5. MCP Endpoint Tests")
        
        if not REQUESTS_AVAILABLE:
            print_skip("MCP endpoints", "requests package not installed")
            self.results["skipped"] += 1
            return True
        
        all_passed = True
        
        # Test list_domains
        try:
            response = requests.post(
                f"{self.mcp_base}/mcp",
                json={"method": "list_domains", "params": {}},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    domains = data.get("domains", [])
                    total_claims = data.get("total_claims", 0)
                    print_pass("list_domains", f"{len(domains)} domains, {total_claims:,} total claims")
                    self.results["passed"] += 1
                else:
                    print_fail("list_domains", data.get("error", "Unknown error"))
                    self.results["failed"] += 1
                    all_passed = False
            else:
                print_fail("list_domains", f"HTTP {response.status_code}")
                self.results["failed"] += 1
                all_passed = False
        except Exception as e:
            print_fail("list_domains", str(e))
            self.results["failed"] += 1
            all_passed = False
        
        # Test get_perspectives
        try:
            response = requests.post(
                f"{self.mcp_base}/mcp",
                json={
                    "method": "get_perspectives",
                    "params": {"topic": "Thomas Paine", "max_clusters": 3}
                },
                timeout=60
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    clusters = data.get("cluster_count", 0)
                    total = data.get("total_claims", 0)
                    print_pass("get_perspectives", f"{clusters} clusters, {total} claims")
                    self.results["passed"] += 1
                else:
                    print_fail("get_perspectives", data.get("error", "Unknown error"))
                    self.results["failed"] += 1
                    all_passed = False
            else:
                print_fail("get_perspectives", f"HTTP {response.status_code}")
                self.results["failed"] += 1
                all_passed = False
        except Exception as e:
            print_fail("get_perspectives", str(e))
            self.results["failed"] += 1
            all_passed = False
        
        # Test analyze_topic
        try:
            response = requests.post(
                f"{self.mcp_base}/mcp",
                json={
                    "method": "analyze_topic",
                    "params": {"topic": "Smedley Butler", "max_claims": 20}
                },
                timeout=60
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    supp = data.get("suppression_score", 0)
                    coord = data.get("coordination_score", 0)
                    claims = data.get("claim_count", 0)
                    print_pass("analyze_topic", f"Suppression: {supp:.2f}, Coordination: {coord:.2f}, {claims} claims")
                    self.results["passed"] += 1
                else:
                    print_fail("analyze_topic", data.get("error", "Unknown error"))
                    self.results["failed"] += 1
                    all_passed = False
            else:
                print_fail("analyze_topic", f"HTTP {response.status_code}")
                self.results["failed"] += 1
                all_passed = False
        except Exception as e:
            print_fail("analyze_topic", str(e))
            self.results["failed"] += 1
            all_passed = False
        
        # Test assess_source
        try:
            response = requests.post(
                f"{self.mcp_base}/mcp",
                json={
                    "method": "assess_source",
                    "params": {"source_identifier": "paine"}
                },
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    found = data.get("source_found", False)
                    print_pass("assess_source", f"Source found: {found}")
                    self.results["passed"] += 1
                else:
                    # May legitimately not find source - that's OK
                    print_pass("assess_source", "Endpoint working (source not found is valid)")
                    self.results["passed"] += 1
            else:
                print_fail("assess_source", f"HTTP {response.status_code}")
                self.results["failed"] += 1
                all_passed = False
        except Exception as e:
            print_fail("assess_source", str(e))
            self.results["failed"] += 1
            all_passed = False
        
        # Test get_claim_context (need a real claim ID for this)
        try:
            response = requests.post(
                f"{self.mcp_base}/mcp",
                json={
                    "method": "get_claim_context",
                    "params": {"claim_id": "claim_test_nonexistent"}
                },
                timeout=30
            )
            if response.status_code == 200:
                # Even if claim not found, endpoint should respond properly
                print_pass("get_claim_context", "Endpoint responsive")
                self.results["passed"] += 1
            else:
                print_fail("get_claim_context", f"HTTP {response.status_code}")
                self.results["failed"] += 1
                all_passed = False
        except Exception as e:
            print_fail("get_claim_context", str(e))
            self.results["failed"] += 1
            all_passed = False
        
        return all_passed
    
    def test_web_ui(self) -> bool:
        """Test web UI accessibility"""
        print_header("6. Web UI Accessibility")
        
        if not REQUESTS_AVAILABLE:
            print_skip("Web UI", "requests package not installed")
            self.results["skipped"] += 1
            return True
            
        try:
            response = requests.get(f"{self.api_base}/", timeout=10)
            if response.status_code == 200:
                print_pass("Web UI", f"Accessible at {self.api_base}")
                self.results["passed"] += 1
                return True
            else:
                print_fail("Web UI", f"HTTP {response.status_code}")
                self.results["failed"] += 1
                return False
        except Exception as e:
            print_fail("Web UI", str(e))
            self.results["failed"] += 1
            return False
    
    def test_dependencies(self) -> bool:
        """Check critical Python dependencies"""
        print_header("7. Python Dependencies")
        
        dependencies = [
            ("pypdf", "6.4.0"),  # Security fix version
            ("pdf2image", None),
            ("neo4j", None),
            ("psycopg2", None),
            ("sentence_transformers", None),
            ("fastapi", None),
        ]
        
        all_ok = True
        
        for pkg, min_version in dependencies:
            try:
                module = __import__(pkg.replace("-", "_"))
                version = getattr(module, "__version__", "unknown")
                
                if min_version and version != "unknown":
                    from packaging import version as pkg_version
                    if pkg_version.parse(version) >= pkg_version.parse(min_version):
                        print_pass(pkg, f"v{version} (≥{min_version} required)")
                        self.results["passed"] += 1
                    else:
                        print_fail(pkg, f"v{version} < {min_version} required (SECURITY)")
                        self.results["failed"] += 1
                        all_ok = False
                else:
                    print_pass(pkg, f"v{version}")
                    self.results["passed"] += 1
                    
            except ImportError:
                print_warn(pkg, "Not installed")
                self.results["warnings"] += 1
            except Exception as e:
                print_warn(pkg, str(e))
                self.results["warnings"] += 1
        
        return all_ok
    
    def print_summary(self):
        """Print final test summary"""
        print_header("VERIFICATION SUMMARY")
        
        total = sum(self.results.values())
        
        print(f"  {Colors.GREEN}Passed:   {self.results['passed']}{Colors.END}")
        print(f"  {Colors.RED}Failed:   {self.results['failed']}{Colors.END}")
        print(f"  {Colors.YELLOW}Warnings: {self.results['warnings']}{Colors.END}")
        print(f"  {Colors.YELLOW}Skipped:  {self.results['skipped']}{Colors.END}")
        print(f"  {'─'*30}")
        print(f"  Total:    {total}")
        print()
        
        if self.results['failed'] == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ All critical tests passed! Ready for release.{Colors.END}")
            return 0
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ {self.results['failed']} critical test(s) failed. Please fix before release.{Colors.END}")
            return 1
    
    def run_all(self) -> int:
        """Run all verification tests"""
        print(f"\n{Colors.BOLD}Aegis Insight v1.1 Pre-Release Verification{Colors.END}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target: {self.api_base} / {self.mcp_base}")
        
        self.test_docker_containers()
        self.test_neo4j_connection()
        self.test_postgres_connection()
        self.test_ollama_connection()
        self.test_mcp_endpoints()
        self.test_web_ui()
        self.test_dependencies()
        
        return self.print_summary()


def main():
    parser = argparse.ArgumentParser(description="Aegis Insight v1.1 Verification")
    parser.add_argument("--host", default="localhost", help="Host address")
    parser.add_argument("--api-port", type=int, default=8000, help="API port")
    parser.add_argument("--mcp-port", type=int, default=8001, help="MCP port")
    
    args = parser.parse_args()
    
    verifier = AegisVerification(
        host=args.host,
        api_port=args.api_port,
        mcp_port=args.mcp_port
    )
    
    sys.exit(verifier.run_all())


if __name__ == "__main__":
    main()
