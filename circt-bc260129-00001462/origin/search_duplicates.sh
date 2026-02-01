#!/bin/bash
set -e

REPO="llvm/circt"

# Extract keywords from analysis.json
KEYWORDS=(
    "concurrent assertion"
    "action block"
    "assert property"
    "else \$error"
    "ImportVerilog"
    "not supported yet"
    "ConcurrentAssertionStatement"
)

echo "========================================"
echo "DUPLICATE SEARCH WORKFLOW"
echo "========================================"
echo ""
echo "Repository: $REPO"
echo "Keywords: ${KEYWORDS[@]}"
echo ""

# Initialize results file
echo "[]" > raw_issues.json

# Search function
search_and_store() {
    local query="$1"
    local label="$2"
    
    echo "[$label] Searching: $query"
    
    result=$(gh issue list -R "$REPO" \
        --search "$query" \
        --limit 20 \
        --json number,title,body,labels,state,url,createdAt 2>&1 || echo "[]")
    
    if [ "$result" != "[]" ] && [ -n "$result" ]; then
        echo "  → Found $(echo "$result" | jq 'length' 2>/dev/null || echo 0) issues"
        echo "$result" | jq -c '.[]' >> raw_issues_list.jsonl
    else
        echo "  → No results"
    fi
}

# Step 1: Search by keywords
echo "Step 1: Searching by keywords..."
search_and_store "concurrent assertion" "keyword1"
search_and_store "action block assertion" "keyword2"
search_and_store "assert property ImportVerilog" "keyword3"
search_and_store "else \$error assertion" "keyword4"

# Step 2: Search by component
echo ""
echo "Step 2: Searching by ImportVerilog..."
search_and_store "ImportVerilog" "component1"

# Step 3: Search by error message
echo ""
echo "Step 3: Searching by error message..."
search_and_store "not supported yet" "error1"

# Step 4: Search by dialect/labels
echo ""
echo "Step 4: Searching by Moore dialect..."
search_and_store "label:Moore" "dialect1"

# Deduplicate by issue number
if [ -f raw_issues_list.jsonl ]; then
    cat raw_issues_list.jsonl | jq -s 'unique_by(.number) | sort_by(-.createdAt)' > unique_issues.json
else
    echo "[]" > unique_issues.json
fi

ISSUE_COUNT=$(jq 'length' unique_issues.json)
echo ""
echo "========================================"
echo "Found $ISSUE_COUNT unique issues"
echo "========================================"
echo ""

