#!/bin/bash

echo "=== Verification Report ==="
echo ""

# Check status.json
echo "1. Status File:"
if [ -f status.json ]; then
    STATUS=$(jq -r '.status' status.json)
    PHASE=$(jq -r '.phase' status.json)
    echo "   ✅ status.json exists"
    echo "   - Status: $STATUS"
    echo "   - Phase: $PHASE"
else
    echo "   ❌ status.json missing"
fi
echo ""

# Check Phase 1 outputs
echo "2. Phase 1 Outputs (Reproduce + Root Cause):"
for file in reproduce.log metadata.json root_cause.md analysis.json; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file missing"
    fi
done
echo ""

# Check Phase 2 outputs
echo "3. Phase 2 Outputs (Minimize + Validate + Duplicates):"
for file in bug.sv error.log command.txt minimize_report.md validation.json validation.md duplicates.json duplicates.md; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file missing"
    fi
done
echo ""

# Check Phase 3 outputs
echo "4. Phase 3 Output (Issue Report):"
if [ -f issue.md ]; then
    echo "   ✅ issue.md"
    LINES=$(wc -l < issue.md)
    echo "   - Lines: $LINES"
else
    echo "   ❌ issue.md missing"
fi
echo ""

# Validate key results
echo "5. Key Results Validation:"
REPRODUCED=$(jq -r '.phase1_results.reproduce.reproduced' status.json 2>/dev/null || echo "N/A")
REDUCTION=$(jq -r '.phase2_results.minimize.reduction_percent' status.json 2>/dev/null || echo "N/A")
VALIDATION=$(jq -r '.phase2_results.validate.result' status.json 2>/dev/null || echo "N/A")
DUPLICATES=$(jq -r '.phase2_results.duplicates.recommendation' status.json 2>/dev/null || echo "N/A")

echo "   - Reproduced: $REPRODUCED"
echo "   - Reduction: $REDUCTION%"
echo "   - Validation: $VALIDATION"
echo "   - Duplicates: $DUPLICATES"
echo ""

echo "6. Issue Report Not Submitted:"
ISSUE_FILE=$(jq -r '.phase3_results.issue_file' status.json 2>/dev/null || echo "N/A")
READY=$(jq -r '.phase3_results.ready_for_submission' status.json 2>/dev/null || echo "N/A")
echo "   ✅ Issue generated in: $ISSUE_FILE"
echo "   ✅ Ready for submission: $READY"
echo "   ⚠️  NOT submitted to GitHub (as requested)"
