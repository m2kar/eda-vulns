#!/bin/bash

REPO="llvm/circt"

echo "Testing gh CLI with different queries..."

# Test 1: Direct search for sim.fmt.literal
echo ""
echo "Search 1: sim.fmt.literal"
gh issue list -R "$REPO" --search "sim.fmt.literal" --limit 5 --json number,title,url 2>&1 | head -20

echo ""
echo "Search 2: arcilator"
gh issue list -R "$REPO" --search "arcilator" --limit 5 --json number,title,url 2>&1 | head -20

echo ""
echo "Search 3: assertion"
gh issue list -R "$REPO" --search "assertion" --limit 5 --json number,title,url 2>&1 | head -20

