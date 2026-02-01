#!/bin/bash

# Extract keywords and search parameters
DIALECT="Sim"
FAILING_PASS="legalization"
CRASH_TYPE="legalization_failure"
ASSERTION_MSG="failed to legalize operation 'sim.fmt.literal'"
KEYWORDS=("sim.fmt.literal" "arcilator" "assertion" "legalize" "sim dialect" "Arc" "lowering")

REPO="llvm/circt"

echo "========================================"
echo "SEARCHING FOR DUPLICATE ISSUES"
echo "========================================"
echo ""
echo "Search Parameters:"
echo "  Dialect: $DIALECT"
echo "  Failing Pass: $FAILING_PASS"
echo "  Crash Type: $CRASH_TYPE"
echo "  Keywords: ${KEYWORDS[@]}"
echo ""

# Initialize results file
> search_results_raw.jsonl

# Search function - using gh CLI
search_github() {
    local query="$1"
    echo "  Searching: $query"
    
    # Search open and closed issues
    gh issue list -R "$REPO" \
        --search "$query" \
        --limit 15 \
        --json number,title,body,labels,state,url \
        2>/dev/null || echo "[]"
}

echo "========================================"
echo "Search 1: Keywords (sim.fmt.literal, arcilator, assertion)"
echo "========================================"
for keyword in "sim.fmt.literal" "arcilator" "assertion" "legalize"; do
    result=$(search_github "$keyword")
    if [ "$result" != "[]" ]; then
        echo "$result" | jq -c '.[]' >> search_results_raw.jsonl
    fi
done

echo ""
echo "========================================"
echo "Search 2: Arc dialect + assertion"
echo "========================================"
search_github "Arc assertion" >> search_results_raw.jsonl 2>/dev/null || true

echo ""
echo "========================================"
echo "Search 3: SimToArc conversion"
echo "========================================"
search_github "SimToArc" >> search_results_raw.jsonl 2>/dev/null || true

echo ""
echo "========================================"
echo "Search 4: Specific issue references"
echo "========================================"
search_github "#6810" >> search_results_raw.jsonl 2>/dev/null || true

# Deduplicate by issue number
echo ""
echo "Deduplicating results..."

if [ -f search_results_raw.jsonl ]; then
    cat search_results_raw.jsonl | jq -s 'unique_by(.number) | sort_by(.number | tonumber) | reverse' > search_results.json
    COUNT=$(jq 'length' search_results.json)
    echo "Found $COUNT unique issues"
else
    echo "[]" > search_results.json
    echo "No results found"
fi

