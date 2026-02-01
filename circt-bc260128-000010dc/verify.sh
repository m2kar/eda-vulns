#!/bin/bash
echo "=== Phase 1 Files ==="
ls -lh reproduce.log metadata.json root_cause.md analysis.json 2>/dev/null | awk '{print $9, $5}' || echo "MISSING"

echo -e "\n=== Phase 2 Files ==="
ls -lh bug.sv error.log command.txt minimize_report.md validation.json validation.md duplicates.json duplicates.md 2>/dev/null | awk '{print $9, $5}' || echo "MISSING"

echo -e "\n=== Phase 3 Files ==="
ls -lh issue.md 2>/dev/null | awk '{print $9, $5}' || echo "MISSING"

echo -e "\n=== File Count ==="
echo -n "Total files: "
find . -maxdepth 1 -type f \( -name "*.log" -o -name "*.json" -o -name "*.md" -o -name "*.txt" -o -name "*.sv" \) | wc -l
