#!/bin/bash

# Verify gh CLI
if ! command -v gh &> /dev/null; then
    echo "ERROR: gh CLI not found"
    exit 1
fi

if ! gh auth status &> /dev/null; then
    echo "ERROR: gh CLI not authenticated"
    exit 1
fi

echo "✓ GitHub CLI ready"
echo ""

# Extract from analysis.json
DIALECT=$(jq -r '.dialect // "unknown"' analysis.json)
FAILING_PASS=$(jq -r '.failing_pass // "unknown"' analysis.json)
CRASH_TYPE=$(jq -r '.crash_type // "unknown"' analysis.json)
TIMEOUT_SECONDS=$(jq -r '.timeout_seconds // 0' analysis.json)
KEYWORDS=$(jq -r '.keywords[]?' analysis.json)

echo "=== Search Parameters ==="
echo "Dialect: $DIALECT"
echo "Failing Pass: $FAILING_PASS"
echo "Crash Type: $CRASH_TYPE"
echo "Timeout: ${TIMEOUT_SECONDS}s"
echo ""
echo "Keywords:"
echo "$KEYWORDS" | sed 's/^/  - /'
echo ""

# Initialize result files
> search_results_raw.jsonl

# Search GitHub Issues
REPO="llvm/circt"

# Helper function to search
search_gh() {
    local query="$1"
    local description="$2"
    
    echo "Searching: $description"
    echo "  Query: $query"
    
    gh issue list -R "$REPO" \
        --search "$query" \
        --limit 15 \
        --json number,title,body,labels,state,url,createdAt,updatedAt \
        2>/dev/null || echo "[]"
}

# Search 1: arcilator + timeout/hang
echo ""
echo "=== SEARCH 1: arcilator timeout/hang ==="
result=$(search_gh "arcilator timeout OR arcilator hang" "arcilator + timeout")
if [ "$result" != "[]" ]; then
    echo "$result" | jq -c '.[]' >> search_results_raw.jsonl
    echo "  Found: $(echo "$result" | jq 'length') issues"
fi

# Search 2: arcilator + struct
echo ""
echo "=== SEARCH 2: arcilator struct ==="
result=$(search_gh "arcilator struct" "arcilator + struct")
if [ "$result" != "[]" ]; then
    echo "$result" | jq -c '.[]' >> search_results_raw.jsonl
    echo "  Found: $(echo "$result" | jq 'length') issues"
fi

# Search 3: struct type coercion/conversion
echo ""
echo "=== SEARCH 3: struct type coercion ==="
result=$(search_gh "struct type coercion OR struct type conversion" "struct + type coercion")
if [ "$result" != "[]" ]; then
    echo "$result" | jq -c '.[]' >> search_results_raw.jsonl
    echo "  Found: $(echo "$result" | jq 'length') issues"
fi

# Search 4: packed struct port
echo ""
echo "=== SEARCH 4: packed struct port ==="
result=$(search_gh "packed struct port" "packed struct + port")
if [ "$result" != "[]" ]; then
    echo "$result" | jq -c '.[]' >> search_results_raw.jsonl
    echo "  Found: $(echo "$result" | jq 'length') issues"
fi

# Search 5: Moore struct
echo ""
echo "=== SEARCH 5: Moore dialect struct ==="
result=$(search_gh "Moore struct" "Moore + struct")
if [ "$result" != "[]" ]; then
    echo "$result" | jq -c '.[]' >> search_results_raw.jsonl
    echo "  Found: $(echo "$result" | jq 'length') issues"
fi

# Search 6: HW dialect implicit conversion
echo ""
echo "=== SEARCH 6: HW implicit conversion ==="
result=$(search_gh "HW dialect implicit conversion OR type coercion" "HW + implicit")
if [ "$result" != "[]" ]; then
    echo "$result" | jq -c '.[]' >> search_results_raw.jsonl
    echo "  Found: $(echo "$result" | jq 'length') issues"
fi

# Deduplicate by issue number
if [ -f search_results_raw.jsonl ]; then
    echo ""
    echo "=== Deduplicating results ==="
    cat search_results_raw.jsonl | jq -s 'unique_by(.number) | sort_by(.number)' > unique_issues.json
    ISSUE_COUNT=$(jq 'length' unique_issues.json)
    echo "Total unique issues found: $ISSUE_COUNT"
else
    echo "[]" > unique_issues.json
    ISSUE_COUNT=0
fi

echo ""
echo "Results saved to unique_issues.json"

