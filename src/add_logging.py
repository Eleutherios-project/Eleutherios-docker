#!/usr/bin/env python3
"""
Add detailed logging to see what record contains
"""

import sys

file_path = "/media/bob/RAID11/DataShare/AegisTrustNet/src/api_graph_extension.py"

with open(file_path, 'r') as f:
    lines = f.readlines()

# Find the line with "record = result.single()" in the get_graph_for_query function
for i, line in enumerate(lines):
    if 'record = result.single()' in line and i < 150:  # First occurrence
        # Add logging right after this line
        indent = '            '  # Match the indentation
        
        new_lines = [
            f"{indent}# DEBUG: Log what record contains\n",
            f"{indent}logger.info(f\"Record keys: {{list(record.keys())}}\")\n",
            f"{indent}if record:\n",
            f"{indent}    logger.info(f\"  claim_rels type: {{type(record['claim_rels'])}}\")\n",
            f"{indent}    logger.info(f\"  claim_rels length: {{len(record['claim_rels']) if record['claim_rels'] else 0}}\")\n",
            f"{indent}    logger.info(f\"  entity_rels type: {{type(record['entity_rels'])}}\")\n",
            f"{indent}    logger.info(f\"  entity_rels length: {{len(record['entity_rels']) if record['entity_rels'] else 0}}\")\n",
            f"{indent}    logger.info(f\"  First claim_rel: {{record['claim_rels'][0] if record['claim_rels'] else 'empty'}}\")\n",
        ]
        
        # Insert after the record = result.single() line
        lines = lines[:i+1] + new_lines + lines[i+1:]
        break

# Write back
with open(file_path, 'w') as f:
    f.writelines(lines)

print("✓ Added detailed logging to api_graph_extension.py")
print("✓ Restart API server to see the logs")
