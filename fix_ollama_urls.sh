#!/bin/bash
# fix_ollama_urls.sh - Patch hardcoded Ollama URLs to use environment variable
# Run from: /media/bob/RAID11/DataShare/Eleutherios_docker_PROD_v1.0/aegis-insight/

set -e

echo "=========================================="
echo "OLLAMA URL Environment Variable Fix"
echo "=========================================="
echo ""

# Files to patch
FILES=(
    "aegis_authority_domain_analyzer.py"
    "aegis_citation_extractor.py"
    "aegis_claim_extractor.py"
    "aegis_coreference_resolver.py"
    "aegis_emotion_extractor.py"
    "aegis_entity_extractor.py"
    "aegis_extraction_orchestrator.py"
    "aegis_import_wizard.py"
)

# Backup directory
BACKUP_DIR="backups/ollama_fix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "✓ Created backup directory: $BACKUP_DIR"
echo ""

# Track changes
CHANGED=0
SKIPPED=0

for FILE in "${FILES[@]}"; do
    if [ ! -f "$FILE" ]; then
        echo "⚠️  SKIP: $FILE (not found)"
        ((SKIPPED++))
        continue
    fi
    
    # Backup
    cp "$FILE" "$BACKUP_DIR/"
    
    # Check if already fixed
    if grep -q "os.getenv.*OLLAMA" "$FILE" 2>/dev/null; then
        echo "✓ ALREADY FIXED: $FILE"
        continue
    fi
    
    # Check if has import os
    if ! grep -q "^import os" "$FILE"; then
        # Add import os after other imports
        sed -i '1s/^/import os\n/' "$FILE"
        echo "  Added 'import os' to $FILE"
    fi
    
    # Fix the __init__ default parameter pattern
    # Change: ollama_url: str = "http://localhost:11434"
    # To: ollama_url: str = None
    sed -i 's/ollama_url: str = "http:\/\/localhost:11434"/ollama_url: str = None/g' "$FILE"
    
    # Fix the assignment pattern
    # Change: self.ollama_url = ollama_url
    # To: self.ollama_url = ollama_url or os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    sed -i "s/self\.ollama_url = ollama_url$/self.ollama_url = ollama_url or os.getenv('OLLAMA_HOST', 'http:\/\/localhost:11434')/g" "$FILE"
    
    # Special handling for aegis_import_wizard.py module-level constants
    if [ "$FILE" == "aegis_import_wizard.py" ]; then
        # Fix: OLLAMA_URL = "http://localhost:11434"
        sed -i "s/^OLLAMA_URL = \"http:\/\/localhost:11434\"/OLLAMA_URL = os.getenv('OLLAMA_HOST', 'http:\/\/localhost:11434')/g" "$FILE"
        # Fix: OLLAMA_SECONDARY_URL if present
        sed -i "s/^OLLAMA_SECONDARY_URL = \"http:\/\/localhost:11435\"/OLLAMA_SECONDARY_URL = os.getenv('OLLAMA_SECONDARY_HOST', 'http:\/\/localhost:11435')/g" "$FILE"
    fi
    
    echo "✓ PATCHED: $FILE"
    ((CHANGED++))
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Files patched: $CHANGED"
echo "Files skipped: $SKIPPED"
echo "Backups in: $BACKUP_DIR"
echo ""

echo "=========================================="
echo "DOCKER-COMPOSE.YML UPDATE"
echo "=========================================="
echo ""
echo "Add this to your api service environment in docker-compose.yml:"
echo ""
echo "  api:"
echo "    environment:"
echo "      - OLLAMA_HOST=http://192.168.1.132:11434"
echo ""
echo "Or if you already have an environment section, just add:"
echo "      - OLLAMA_HOST=http://192.168.1.132:11434"
echo ""

echo "=========================================="
echo "VERIFICATION"
echo "=========================================="
echo ""
echo "After bouncing the container, verify with:"
echo "  docker exec aegis-api env | grep OLLAMA"
echo ""
echo "Should show: OLLAMA_HOST=http://192.168.1.132:11434"
echo ""
