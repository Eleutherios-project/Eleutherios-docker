#!/usr/bin/env python3
"""
Aegis Insight - Detection System Builder

Unified infrastructure for building and managing all detection systems:
- Suppression Detector
- Coordination Detector  
- Anomaly Detector

Features:
- Job orchestration with checkpointing
- Status tracking
- Progress reporting
- Incremental builds
- Failure recovery
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, asdict
import sqlite3
from neo4j import GraphDatabase
from aegis_config import Config


class JobStatus(Enum):
    """Job execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class JobDefinition:
    """Definition of a detection setup job"""
    job_id: str
    name: str
    description: str
    duration_estimate: str
    required_for: List[str]  # Which detectors need this
    func: Callable
    incremental: bool = True
    critical: bool = True  # If fails, stop pipeline


@dataclass
class JobState:
    """Current state of a job"""
    job_id: str
    status: JobStatus
    progress: float  # 0.0 to 1.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    checkpoint_data: Optional[Dict] = None
    result: Optional[Dict] = None


class DetectionSystemBuilder:
    """
    Unified builder for all detection infrastructure
    
    Manages:
    - Citation network building
    - Embedding generation
    - Temporal indexing
    - Geographic indexing
    - System status tracking
    """
    
    # Job definitions
    JOBS = {
        'build_citations': JobDefinition(
            job_id='build_citations',
            name='Build Citation Network',
            description='Create CITES relationships between claims',
            duration_estimate='15-40 min',
            required_for=['suppression'],
            func=None,  # Set dynamically
            incremental=True,
            critical=True
        ),
        'generate_embeddings': JobDefinition(
            job_id='generate_embeddings',
            name='Generate Embeddings',
            description='Create vector embeddings for claims',
            duration_estimate='10-15 min',
            required_for=['suppression', 'coordination', 'anomaly'],
            func=None,
            incremental=True,
            critical=True
        ),
        'build_temporal_index': JobDefinition(
            job_id='build_temporal_index',
            name='Build Temporal Index',
            description='Index claims by time for coordination detection',
            duration_estimate='5-10 min',
            required_for=['coordination'],
            func=None,
            incremental=True,
            critical=False
        ),
        'build_geographic_index': JobDefinition(
            job_id='build_geographic_index',
            name='Build Geographic Index',
            description='Index claims by location for anomaly detection',
            duration_estimate='5-10 min',
            required_for=['anomaly'],
            func=None,
            incremental=True,
            critical=False
        )
    }
    
    def __init__(self,
                 neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = None,
                 state_db_path: str = "/tmp/aegis_detection_state.db",
                 checkpoint_dir: str = "/tmp/aegis_checkpoints",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize detection system builder
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            state_db_path: Path to SQLite state database
            checkpoint_dir: Directory for checkpoint files
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Neo4j connection
        # Use config fallback if password not provided
        neo4j_password = neo4j_password or Config.NEO4J_PASSWORD
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )
        self.driver.verify_connectivity()
        self.logger.info("✓ Connected to Neo4j")
        
        # State database
        self.state_db_path = state_db_path
        self._init_state_db()
        
        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress callback (for UI updates)
        self.progress_callback: Optional[Callable] = None
    
    def _init_state_db(self):
        """Initialize SQLite database for state tracking"""
        conn = sqlite3.connect(self.state_db_path)
        cursor = conn.cursor()
        
        # Job status table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_status (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress REAL DEFAULT 0.0,
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                checkpoint_data TEXT,
                result TEXT
            )
        """)
        
        # System status table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_status (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """)
        
        # Batch tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_ingestion (
                batch_id TEXT PRIMARY KEY,
                ingested_at TEXT,
                needs_detection_setup INTEGER DEFAULT 1,
                detection_setup_at TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        
        self.logger.info("✓ State database initialized")
    
    def get_system_status(self) -> Dict:
        """
        Get current system status
        
        Returns dict with:
        - claims_count: Total claims in graph
        - citations_count: Total CITES relationships
        - embeddings_count: Claims with embeddings
        - detectors_ready: Dict of detector readiness
        - needs_action: List of jobs that need running
        """
        with self.driver.session() as session:
            # Count claims
            result = session.run("MATCH (c:Claim) RETURN count(c) as count")
            claims_count = result.single()['count']
            
            # Count citations
            try:
                result = session.run("MATCH ()-[r:CITES]->() RETURN count(r) as count")
                citations_count = result.single()['count']
            except:
                citations_count = 0
            
            # Check embeddings (from PostgreSQL or Neo4j property)
            # For now, assume embeddings are in separate system
            embeddings_count = self._get_embeddings_count()
        
        # Check detector readiness
        detectors_ready = {
            'suppression': self._check_detector_ready('suppression'),
            'coordination': self._check_detector_ready('coordination'),
            'anomaly': self._check_detector_ready('anomaly')
        }
        
        # Calculate what needs doing
        needs_action = self._calculate_needs_action(
            claims_count, citations_count, embeddings_count
        )
        
        return {
            'claims_count': claims_count,
            'citations_count': citations_count,
            'embeddings_count': embeddings_count,
            'detectors_ready': detectors_ready,
            'needs_action': needs_action,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_embeddings_count(self) -> int:
        """Get count of embeddings from PostgreSQL or job state"""
        
        # First try PostgreSQL
        try:
            import psycopg2
            
            conn = psycopg2.connect(
                host=os.environ.get("POSTGRES_HOST", "localhost"),
                database="aegis-postgres",
                user="aegis",
                password="aegistrusted",
                connect_timeout=3
            )
            
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM claim_embeddings")
                count = cur.fetchone()[0]
            
            conn.close()
            self.logger.debug(f"Found {count} embeddings in PostgreSQL")
            return count
            
        except Exception as e:
            # PostgreSQL not available - check job completion status
            self.logger.debug(f"PostgreSQL unavailable: {e}")
            
            # Use job state as fallback
            state = self._get_job_state('generate_embeddings')
            if state.status == JobStatus.COMPLETE and state.result:
                count = state.result.get('embeddings_generated', 0)
                self.logger.debug(f"Using job state embedding count: {count}")
                return count
            
            return 0
    
    def _check_detector_ready(self, detector_name: str) -> Dict:
        """
        Check if detector is ready to use
        
        Returns:
            {'ready': bool, 'missing': [job_ids], 'status': str}
        """
        # Get jobs required for this detector
        required_jobs = [
            job_id for job_id, job in self.JOBS.items()
            if detector_name in job.required_for
        ]
        
        # Check status of each job
        missing = []
        for job_id in required_jobs:
            state = self._get_job_state(job_id)
            if state.status != JobStatus.COMPLETE:
                missing.append(job_id)
        
        if not missing:
            return {
                'ready': True,
                'missing': [],
                'status': 'operational'
            }
        else:
            return {
                'ready': False,
                'missing': missing,
                'status': 'needs_setup'
            }
    
    def _calculate_needs_action(self, 
                                claims_count: int,
                                citations_count: int,
                                embeddings_count: int) -> List[Dict]:
        """Calculate what actions are needed"""
        needs = []
        
        if claims_count == 0:
            needs.append({
                'priority': 'critical',
                'action': 'upload_documents',
                'message': 'No data loaded - upload documents to begin'
            })
            return needs
        
        if citations_count == 0:
            needs.append({
                'priority': 'high',
                'action': 'build_citations',
                'message': f'{claims_count} claims need citation network',
                'job_id': 'build_citations'
            })
        
        # Check if embeddings job is complete (don't just rely on count)
        embeddings_job = self._get_job_state('generate_embeddings')
        if embeddings_count == 0 and embeddings_job.status != JobStatus.COMPLETE:
            needs.append({
                'priority': 'high',
                'action': 'generate_embeddings',
                'message': f'{claims_count} claims need embeddings',
                'job_id': 'generate_embeddings'
            })
        
        # Check if batch needs detection setup
        if self._has_pending_batch():
            needs.append({
                'priority': 'medium',
                'action': 'run_detection_setup',
                'message': 'New batch ingested - run detection setup',
                'job_ids': ['build_citations', 'generate_embeddings']
            })
        
        return needs
    
    def _has_pending_batch(self) -> bool:
        """Check if there's a batch waiting for detection setup"""
        conn = sqlite3.connect(self.state_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM batch_ingestion 
            WHERE needs_detection_setup = 1
        """)
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    
    def mark_embeddings_complete(self, count: int):
        """
        Manually mark embeddings as complete
        
        Use this if embeddings were generated but PostgreSQL is not running
        or if the count detection is failing.
        
        Args:
            count: Number of embeddings that were generated
        """
        self._update_job_state(
            'generate_embeddings',
            JobStatus.COMPLETE,
            progress=1.0,
            result={'embeddings_generated': count, 'manual': True}
        )
        self.logger.info(f"✓ Marked {count} embeddings as complete")
    
    def mark_batch_ingested(self, batch_id: str):
        """Mark that a batch has been ingested (called by CLI)"""
        conn = sqlite3.connect(self.state_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO batch_ingestion 
            (batch_id, ingested_at, needs_detection_setup)
            VALUES (?, ?, 1)
        """, (batch_id, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        self.logger.info(f"✓ Marked batch {batch_id} as ingested")
    
    def run_detection_setup(self, 
                           for_detector: Optional[str] = None,
                           skip_completed: bool = True) -> Dict:
        """
        Run detection setup - builds all required infrastructure
        
        Args:
            for_detector: If specified, only build requirements for this detector
                         Options: 'suppression', 'coordination', 'anomaly', or None for all
            skip_completed: Skip jobs that are already complete
        
        Returns:
            Dict with results for each job
        """
        self.logger.info("="*80)
        self.logger.info("DETECTION SETUP STARTING")
        self.logger.info("="*80)
        
        # Determine which jobs to run
        if for_detector:
            jobs_to_run = [
                job_id for job_id, job in self.JOBS.items()
                if for_detector in job.required_for
            ]
            self.logger.info(f"Building requirements for {for_detector} detector")
        else:
            jobs_to_run = list(self.JOBS.keys())
            self.logger.info("Building all detection requirements")
        
        results = {}
        
        for job_id in jobs_to_run:
            job = self.JOBS[job_id]
            
            # Check if already complete
            if skip_completed:
                state = self._get_job_state(job_id)
                if state.status == JobStatus.COMPLETE:
                    self.logger.info(f"⊘ Skipping {job.name} (already complete)")
                    results[job_id] = 'skipped'
                    continue
            
            # Run job
            self.logger.info(f"\n▶ Running: {job.name}")
            self.logger.info(f"  Estimated time: {job.duration_estimate}")
            
            try:
                result = self._run_job_with_checkpoint(job)
                results[job_id] = 'success'
                self.logger.info(f"✓ {job.name} complete")
            except Exception as e:
                results[job_id] = f'failed: {str(e)}'
                self.logger.error(f"✗ {job.name} failed: {e}")
                
                # If critical job fails, stop
                if job.critical:
                    self.logger.error("Critical job failed - stopping pipeline")
                    break
        
        # Mark batch as processed
        self._mark_detection_setup_complete()
        
        self.logger.info("\n" + "="*80)
        self.logger.info("DETECTION SETUP COMPLETE")
        self.logger.info("="*80)
        
        return results
    
    def _run_job_with_checkpoint(self, job: JobDefinition) -> Any:
        """
        Run a job with checkpoint support
        
        Handles:
        - Loading previous checkpoint if exists
        - Running job function
        - Saving checkpoints during execution
        - Updating job state
        - Error handling
        """
        job_id = job.job_id
        
        # Load checkpoint if exists
        checkpoint = self._load_checkpoint(job_id)
        
        # Update job state to running
        self._update_job_state(job_id, JobStatus.RUNNING, progress=0.0)
        
        try:
            # Run job function with checkpoint and progress callback
            result = job.func(
                checkpoint=checkpoint,
                progress_callback=lambda p: self._update_progress(job_id, p),
                builder=self  # Pass self for access to Neo4j, etc.
            )
            
            # Mark complete
            self._update_job_state(
                job_id, 
                JobStatus.COMPLETE, 
                progress=1.0,
                result=result
            )
            
            # Clear checkpoint
            self._clear_checkpoint(job_id)
            
            return result
            
        except Exception as e:
            # Save checkpoint on failure
            self._save_checkpoint(job_id, checkpoint)
            
            # Mark failed
            self._update_job_state(
                job_id,
                JobStatus.FAILED,
                error_message=str(e)
            )
            
            raise
    
    def _update_progress(self, job_id: str, progress: float):
        """Update job progress"""
        self._update_job_state(job_id, JobStatus.RUNNING, progress=progress)
        
        # Call external progress callback if set
        if self.progress_callback:
            self.progress_callback(job_id, progress)
    
    def _get_job_state(self, job_id: str) -> JobState:
        """Get current state of a job"""
        conn = sqlite3.connect(self.state_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT status, progress, started_at, completed_at, 
                   error_message, checkpoint_data, result
            FROM job_status WHERE job_id = ?
        """, (job_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return JobState(
                job_id=job_id,
                status=JobStatus.PENDING,
                progress=0.0
            )
        
        return JobState(
            job_id=job_id,
            status=JobStatus(row[0]),
            progress=row[1],
            started_at=datetime.fromisoformat(row[2]) if row[2] else None,
            completed_at=datetime.fromisoformat(row[3]) if row[3] else None,
            error_message=row[4],
            checkpoint_data=json.loads(row[5]) if row[5] else None,
            result=json.loads(row[6]) if row[6] else None
        )
    
    def _update_job_state(self,
                         job_id: str,
                         status: JobStatus,
                         progress: Optional[float] = None,
                         error_message: Optional[str] = None,
                         result: Optional[Dict] = None):
        """Update job state in database"""
        conn = sqlite3.connect(self.state_db_path)
        cursor = conn.cursor()
        
        # Get current state
        cursor.execute("SELECT * FROM job_status WHERE job_id = ?", (job_id,))
        exists = cursor.fetchone() is not None
        
        now = datetime.now().isoformat()
        
        if exists:
            # Update existing
            updates = ["status = ?"]
            values = [status.value]
            
            if progress is not None:
                updates.append("progress = ?")
                values.append(progress)
            
            if error_message is not None:
                updates.append("error_message = ?")
                values.append(error_message)
            
            if result is not None:
                updates.append("result = ?")
                values.append(json.dumps(result))
            
            if status == JobStatus.RUNNING:
                updates.append("started_at = ?")
                values.append(now)
            
            if status == JobStatus.COMPLETE:
                updates.append("completed_at = ?")
                values.append(now)
            
            values.append(job_id)
            
            cursor.execute(f"""
                UPDATE job_status 
                SET {', '.join(updates)}
                WHERE job_id = ?
            """, values)
        else:
            # Insert new
            cursor.execute("""
                INSERT INTO job_status 
                (job_id, status, progress, started_at)
                VALUES (?, ?, ?, ?)
            """, (job_id, status.value, progress or 0.0, now))
        
        conn.commit()
        conn.close()
    
    def _load_checkpoint(self, job_id: str) -> Optional[Dict]:
        """Load checkpoint data for a job"""
        checkpoint_file = self.checkpoint_dir / f"{job_id}.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        return None
    
    def _save_checkpoint(self, job_id: str, data: Dict):
        """Save checkpoint data for a job"""
        checkpoint_file = self.checkpoint_dir / f"{job_id}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f)
    
    def _clear_checkpoint(self, job_id: str):
        """Clear checkpoint for a job"""
        checkpoint_file = self.checkpoint_dir / f"{job_id}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
    
    def _mark_detection_setup_complete(self):
        """Mark that detection setup is complete for pending batches"""
        conn = sqlite3.connect(self.state_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE batch_ingestion 
            SET needs_detection_setup = 0,
                detection_setup_at = ?
            WHERE needs_detection_setup = 1
        """, (datetime.now().isoformat(),))
        conn.commit()
        conn.close()
    
    def close(self):
        """Close connections"""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed")

    def check_neo4j_connection(self) -> bool:
        """Check if Neo4j is connected"""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            return True
        except:
            return False

    def check_ollama_availability(self) -> bool:
        """Check if Ollama is available"""
        try:
            import requests
            response = requests.get(f"{os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def get_database_size(self) -> str:
        """Get database size estimate"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (n)
                    RETURN count(n) as nodes
                """)
                nodes = result.single()['nodes']

                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN count(r) as rels
                """)
                rels = result.single()['rels']

                # Rough size estimate
                size_mb = (nodes * 1 + rels * 0.5) / 1000  # Very rough!

                if size_mb < 1:
                    return f"{size_mb * 1000:.0f} KB"
                elif size_mb < 1000:
                    return f"{size_mb:.1f} MB"
                else:
                    return f"{size_mb / 1000:.1f} GB"
        except:
            return "Unknown"

    def get_last_ingestion_time(self) -> Optional[str]:
        """Get timestamp of last data ingestion"""
        try:
            # Check state database for last ingestion
            cursor = self.state_conn.cursor()
            cursor.execute("""
                SELECT value FROM system_status 
                WHERE key = 'last_ingestion'
            """)
            result = cursor.fetchone()
            return result[0] if result else None
        except:
            return None

    def get_active_jobs(self) -> dict:
        """Get currently running jobs with progress"""
        try:
            cursor = self.state_conn.cursor()
            cursor.execute("""
                SELECT job_id, status, progress, started_at, updated_at
                FROM job_status
                WHERE status = 'running'
            """)

            jobs = {}
            for row in cursor.fetchall():
                job_id = row[0]
                jobs[job_id] = {
                    'status': row[1],
                    'progress': row[2],
                    'started_at': row[3],
                    'updated_at': row[4],
                    'eta': self._calculate_eta(row[2], row[3], row[4])
                }

            return jobs
        except:
            return {}

    def _calculate_eta(self, progress: float, started_at: str, updated_at: str) -> str:
        """Calculate estimated time remaining"""
        try:
            from datetime import datetime

            if progress <= 0:
                return "Calculating..."

            started = datetime.fromisoformat(started_at)
            updated = datetime.fromisoformat(updated_at)

            elapsed = (updated - started).total_seconds()
            rate = progress / elapsed if elapsed > 0 else 0

            if rate <= 0:
                return "Calculating..."

            remaining = (1.0 - progress) / rate

            if remaining < 60:
                return f"{int(remaining)}s"
            elif remaining < 3600:
                return f"{int(remaining / 60)}m"
            else:
                return f"{int(remaining / 3600)}h {int((remaining % 3600) / 60)}m"
        except:
            return "Unknown"

    def get_detector_requirements(self, detector: str) -> dict:
        """Get required jobs for a specific detector"""
        requirements_map = {
            'suppression': ['build_citations'],
            'coordination': ['generate_embeddings', 'build_temporal_index'],
            'anomaly': ['generate_embeddings', 'build_geographic_index']
        }

        required = requirements_map.get(detector, [])

        # Check which are missing
        status = self.get_system_status()
        detector_status = status['detectors_ready'].get(detector, {})

        return {
            'required': required,
            'missing': detector_status.get('missing', [])
        }

def main():
    """Test detection system builder"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    builder = DetectionSystemBuilder()
    
    # Check system status
    print("\n" + "="*80)
    print("SYSTEM STATUS")
    print("="*80)
    
    status = builder.get_system_status()
    
    print(f"\nClaims: {status['claims_count']}")
    print(f"Citations: {status['citations_count']}")
    print(f"Embeddings: {status['embeddings_count']}")
    
    print("\nDetector Readiness:")
    for detector, state in status['detectors_ready'].items():
        status_str = "✓ Ready" if state['ready'] else f"⚠️  Missing: {', '.join(state['missing'])}"
        print(f"  {detector}: {status_str}")
    
    print("\nActions Needed:")
    if status['needs_action']:
        for action in status['needs_action']:
            print(f"  [{action['priority']}] {action['message']}")
    else:
        print("  None - system ready!")
    
    builder.close()


if __name__ == "__main__":
    main()
