#!/usr/bin/env python3
"""
Aegis Insight - Ollama URL Hardcode Fixer
=========================================

This script patches all hardcoded Ollama localhost URLs to use the 
OLLAMA_HOST environment variable with localhost as fallback.

Usage:
    python fix_ollama_urls.py                    # Dry run (preview changes)
    python fix_ollama_urls.py --apply            # Apply changes
    python fix_ollama_urls.py --apply --no-backup # Apply without backups

Author: Aegis Insight Team
Date: January 2026
"""

import os
import re
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

# Configuration
BACKUP_SUFFIX = f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOCALHOST_URL = "http://localhost:11434"
ENV_VAR_NAME = "OLLAMA_HOST"


class OllamaURLFixer:
    """Fixes hardcoded Ollama URLs across the codebase."""
    
    def __init__(self, base_dir: str, dry_run: bool = True, create_backups: bool = True):
        self.base_dir = Path(base_dir)
        self.dry_run = dry_run
        self.create_backups = create_backups
        self.changes: List[Dict] = []
        self.errors: List[str] = []
        
    def run(self) -> bool:
        """Execute the patching process."""
        print("=" * 70)
        print("AEGIS INSIGHT - Ollama URL Hardcode Fixer")
        print("=" * 70)
        print(f"Base directory: {self.base_dir}")
        print(f"Mode: {'DRY RUN (preview only)' if self.dry_run else 'APPLYING CHANGES'}")
        print(f"Backups: {'Yes' if self.create_backups and not self.dry_run else 'No'}")
        print("=" * 70)
        print()
        
        # Define all files and their specific fixes
        fixes = self._get_fix_definitions()
        
        for fix in fixes:
            self._process_file(fix)
        
        self._print_summary()
        return len(self.errors) == 0
    
    def _get_fix_definitions(self) -> List[Dict]:
        """Define all files and the specific fixes needed."""
        return [
            # Extractors with function parameter defaults
            {
                'file': 'aegis_citation_extractor.py',
                'patterns': [
                    (r'ollama_url: str = "http://localhost:11434"',
                     'ollama_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")'),
                ],
                'needs_os_import': True,
            },
            {
                'file': 'aegis_claim_extractor.py',
                'patterns': [
                    (r'ollama_url: str = "http://localhost:11434"',
                     'ollama_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")'),
                ],
                'needs_os_import': True,
            },
            {
                'file': 'aegis_coreference_resolver.py',
                'patterns': [
                    (r'ollama_url: str = "http://localhost:11434"',
                     'ollama_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")'),
                ],
                'needs_os_import': True,
            },
            {
                'file': 'aegis_emotion_extractor.py',
                'patterns': [
                    (r'ollama_url: str = "http://localhost:11434"',
                     'ollama_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")'),
                ],
                'needs_os_import': True,
            },
            {
                'file': 'aegis_entity_extractor.py',
                'patterns': [
                    (r'ollama_url: str = "http://localhost:11434"',
                     'ollama_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")'),
                ],
                'needs_os_import': True,
            },
            {
                'file': 'aegis_geographic_extractor.py',
                'patterns': [
                    (r'ollama_url: str = "http://localhost:11434"',
                     'ollama_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")'),
                ],
                'needs_os_import': True,
            },
            {
                'file': 'aegis_temporal_extractor.py',
                'patterns': [
                    (r'ollama_url: str = "http://localhost:11434"',
                     'ollama_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")'),
                ],
                'needs_os_import': True,
            },
            {
                'file': 'aegis_extraction_orchestrator.py',
                'patterns': [
                    (r'ollama_url: str = "http://localhost:11434"',
                     'ollama_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")'),
                ],
                'needs_os_import': True,
            },
            {
                'file': 'pattern_search_llm.py',
                'patterns': [
                    (r'ollama_url: str = "http://localhost:11434"',
                     'ollama_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")'),
                ],
                'needs_os_import': True,
            },
            # V3 orchestrator - dataclass field
            {
                'file': 'aegis_extraction_orchestrator_v3.py',
                'patterns': [
                    (r'ollama_url: str = "http://localhost:11434"',
                     'ollama_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")'),
                ],
                'needs_os_import': True,
            },
            # Import wizard - constant
            {
                'file': 'aegis_import_wizard.py',
                'patterns': [
                    (r'OLLAMA_URL = "http://localhost:11434"',
                     'OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")'),
                ],
                'needs_os_import': True,
            },
            # Enhanced claim extractor - two patterns
            {
                'file': 'enhanced_claim_extractor.py',
                'patterns': [
                    (r'ollama_url: str = "http://localhost:11434"',
                     'ollama_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")'),
                    (r'def __init__\(self, base_url: str = "http://localhost:11434"\)',
                     'def __init__(self, base_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434"))'),
                ],
                'needs_os_import': True,
            },
            # Run extraction pipeline - argparse
            {
                'file': 'run_extraction_pipeline.py',
                'patterns': [
                    (r"default='http://localhost:11434'",
                     "default=os.environ.get('OLLAMA_HOST', 'http://localhost:11434')"),
                ],
                'needs_os_import': True,
            },
            # API integration - inline URL
            {
                'file': 'src/api_integration.py',
                'patterns': [
                    (r"'http://localhost:11434/api/generate'",
                     "os.getenv('OLLAMA_HOST', 'http://localhost:11434') + '/api/generate'"),
                ],
                'needs_os_import': True,
            },
            # MCP server - normalize env var name
            {
                'file': 'aegis_mcp_server.py',
                'patterns': [
                    (r'os\.getenv\("OLLAMA_URL"',
                     'os.getenv("OLLAMA_HOST"'),
                ],
                'needs_os_import': False,  # Already has os import
            },
            # Suppression detector v2 - normalize env var name
            {
                'file': 'aegis_suppression_detector_v2.py',
                'patterns': [
                    (r"os\.environ\.get\('OLLAMA_URL'",
                     "os.environ.get('OLLAMA_HOST'"),
                ],
                'needs_os_import': False,  # Already has os import
            },
        ]
    
    def _process_file(self, fix: Dict):
        """Process a single file with its fixes."""
        filepath = self.base_dir / fix['file']
        
        if not filepath.exists():
            self.errors.append(f"File not found: {filepath}")
            print(f"⚠️  SKIP: {fix['file']} (not found)")
            return
        
        try:
            content = filepath.read_text()
            original_content = content
            changes_made = []
            
            # Check if os import is needed and missing
            if fix.get('needs_os_import') and not self._has_os_import(content):
                content = self._add_os_import(content)
                changes_made.append("Added 'import os'")
            
            # Apply pattern replacements
            for pattern, replacement in fix['patterns']:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    changes_made.append(f"Replaced: {pattern[:50]}...")
            
            if changes_made:
                self.changes.append({
                    'file': fix['file'],
                    'changes': changes_made,
                })
                
                print(f"✓  {fix['file']}")
                for change in changes_made:
                    print(f"   └─ {change}")
                
                if not self.dry_run:
                    # Create backup
                    if self.create_backups:
                        backup_path = filepath.with_suffix(filepath.suffix + BACKUP_SUFFIX)
                        shutil.copy2(filepath, backup_path)
                    
                    # Write changes
                    filepath.write_text(content)
            else:
                print(f"○  {fix['file']} (no changes needed)")
                
        except Exception as e:
            self.errors.append(f"Error processing {fix['file']}: {e}")
            print(f"✗  {fix['file']}: {e}")
    
    def _has_os_import(self, content: str) -> bool:
        """Check if 'import os' or 'from os import' exists."""
        # Match various forms of os import
        patterns = [
            r'^import os\s*$',
            r'^import os\s*#',
            r'^import os,',
            r'^from os import',
            r'^import os\s+',
        ]
        for pattern in patterns:
            if re.search(pattern, content, re.MULTILINE):
                return True
        return False
    
    def _add_os_import(self, content: str) -> str:
        """Add 'import os' after existing imports."""
        lines = content.split('\n')
        
        # Find the best place to insert (after other imports)
        insert_idx = 0
        in_docstring = False
        docstring_char = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Track docstrings
            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    docstring_char = stripped[:3]
                    if stripped.count(docstring_char) == 1:
                        in_docstring = True
                    insert_idx = i + 1
                    continue
            else:
                if docstring_char in stripped:
                    in_docstring = False
                insert_idx = i + 1
                continue
            
            # Track imports
            if stripped.startswith('import ') or stripped.startswith('from '):
                insert_idx = i + 1
            elif stripped and not stripped.startswith('#') and insert_idx > 0:
                # Hit non-import code, stop looking
                break
        
        # Insert the import
        lines.insert(insert_idx, 'import os')
        return '\n'.join(lines)
    
    def _print_summary(self):
        """Print summary of changes."""
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Files modified: {len(self.changes)}")
        print(f"Errors: {len(self.errors)}")
        
        if self.dry_run and self.changes:
            print()
            print("This was a DRY RUN. To apply changes, run:")
            print(f"  python {sys.argv[0]} --apply")
        
        if self.errors:
            print()
            print("ERRORS:")
            for error in self.errors:
                print(f"  - {error}")
        
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Fix hardcoded Ollama URLs in Aegis Insight codebase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fix_ollama_urls.py                     # Preview changes (dry run)
  python fix_ollama_urls.py --apply             # Apply changes with backups
  python fix_ollama_urls.py --apply --no-backup # Apply without backups
  python fix_ollama_urls.py --dir /path/to/code # Specify different directory
        """
    )
    parser.add_argument(
        '--apply', 
        action='store_true',
        help='Actually apply the changes (default is dry run)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true', 
        help='Skip creating backup files'
    )
    parser.add_argument(
        '--dir',
        type=str,
        default='.',
        help='Base directory of Aegis Insight code (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Resolve directory
    base_dir = Path(args.dir).resolve()
    
    # Verify it looks like the aegis-insight directory
    expected_files = ['aegis_claim_extractor.py', 'aegis_entity_extractor.py', 'api_server.py']
    missing = [f for f in expected_files if not (base_dir / f).exists()]
    
    if missing:
        print(f"Warning: Expected files not found in {base_dir}:")
        for f in missing:
            print(f"  - {f}")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(1)
    
    fixer = OllamaURLFixer(
        base_dir=base_dir,
        dry_run=not args.apply,
        create_backups=not args.no_backup
    )
    
    success = fixer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
