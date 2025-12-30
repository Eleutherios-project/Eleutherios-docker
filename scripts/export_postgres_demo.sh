#!/bin/bash
#
# PostgreSQL Demo Data Exporter
# =============================
#
# Exports PostgreSQL data to a portable SQL file with proper encoding.
#
# Usage:
#     ./export_postgres_demo.sh [output_dir]
#

set -e

# Configuration
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-aegis-postgres}"
POSTGRES_USER="${POSTGRES_USER:-aegis}"
POSTGRES_DB="${POSTGRES_DB:-aegis_insight}"
OUTPUT_DIR="${1:-demo-data}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "PostgreSQL Demo Data Export"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${POSTGRES_CONTAINER}$"; then
    echo -e "${RED}ERROR: Container $POSTGRES_CONTAINER is not running${NC}"
    echo "Start with: docker-compose up -d"
    exit 1
fi

echo -e "${GREEN}✓ Container running: $POSTGRES_CONTAINER${NC}"

# Get row counts before export
echo ""
echo "Checking table sizes..."
docker exec "$POSTGRES_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
SELECT 
    schemaname || '.' || relname AS table_name,
    n_live_tup AS row_count
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;
"

# Export with proper settings
echo ""
echo "Exporting database..."

OUTPUT_FILE="$OUTPUT_DIR/aegis_insight.sql"
GZIP_FILE="$OUTPUT_DIR/aegis_insight.sql.gz"

# Export options:
# --no-owner          : Don't include ownership commands
# --no-privileges     : Don't include GRANT/REVOKE
# --clean             : Include DROP commands before CREATE
# --if-exists         : Add IF EXISTS to DROP commands
# --encoding=UTF8     : Force UTF-8 encoding
# --column-inserts    : Use INSERT with column names (more portable)
# --rows-per-insert=1000: Batch inserts for performance

docker exec "$POSTGRES_CONTAINER" pg_dump \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" \
    --no-owner \
    --no-privileges \
    --clean \
    --if-exists \
    --encoding=UTF8 \
    --column-inserts \
    --rows-per-insert=1000 \
    > "$OUTPUT_FILE"

# Check export succeeded
if [ ! -s "$OUTPUT_FILE" ]; then
    echo -e "${RED}ERROR: Export file is empty${NC}"
    exit 1
fi

# Get file size
FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
echo -e "${GREEN}✓ Exported: $OUTPUT_FILE ($FILE_SIZE)${NC}"

# Compress
echo "Compressing..."
gzip -f "$OUTPUT_FILE"
GZIP_SIZE=$(du -h "$GZIP_FILE" | cut -f1)
echo -e "${GREEN}✓ Compressed: $GZIP_FILE ($GZIP_SIZE)${NC}"

# Verify the export by checking line count
LINE_COUNT=$(zcat "$GZIP_FILE" | wc -l)
echo ""
echo "Export statistics:"
echo "  Lines: $LINE_COUNT"
echo "  Compressed size: $GZIP_SIZE"

# Create a manifest file
MANIFEST_FILE="$OUTPUT_DIR/postgres_manifest.txt"
echo "PostgreSQL Export Manifest" > "$MANIFEST_FILE"
echo "==========================" >> "$MANIFEST_FILE"
echo "" >> "$MANIFEST_FILE"
echo "Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")" >> "$MANIFEST_FILE"
echo "Database: $POSTGRES_DB" >> "$MANIFEST_FILE"
echo "Container: $POSTGRES_CONTAINER" >> "$MANIFEST_FILE"
echo "Lines: $LINE_COUNT" >> "$MANIFEST_FILE"
echo "Size: $GZIP_SIZE" >> "$MANIFEST_FILE"
echo "" >> "$MANIFEST_FILE"
echo "Tables:" >> "$MANIFEST_FILE"
docker exec "$POSTGRES_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "
SELECT '  - ' || relname || ': ' || n_live_tup || ' rows'
FROM pg_stat_user_tables
ORDER BY relname;
" >> "$MANIFEST_FILE"

echo ""
echo -e "${GREEN}=========================================="
echo "Export complete!"
echo "==========================================${NC}"
echo ""
echo "Output files:"
echo "  $GZIP_FILE"
echo "  $MANIFEST_FILE"
echo ""
echo "To import on target system:"
echo "  gunzip -c $GZIP_FILE | docker exec -i aegis-postgres psql -U aegis -d aegis_insight"
