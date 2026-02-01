#!/bin/bash

set -e

REPO="llvm/circt"
OUTPUT_DIR="."

# Extract search terms from analysis.json
DIALECT="moore"
FAILING_PASS="convert-moore-to-core"
CRASH_TYPE="assertion"
ASSERTION_MSG="dyn_cast on a non-existent value"

# Keywords from analysis.json
KEYWORDS=(
    "StringType"
    "DynamicStringType"
    "getModulePortInfo"
    "sanitizeInOut"
    "dyn_cast non-existent"
    "convertType null"
    "MooreToCore"
    "string output port"
    "type conversion"
    "ModulePortInfo"
)

echo "======================================"
echo "DUPLICATE CHECK WORKFLOW"
echo "======================================"
echo ""
echo "Search Terms:"
echo "- Dialect: $DIALECT"
echo "- Failing Pass: $FAILING_PASS"
echo "- Crash Type: $CRASH_TYPE"
echo ""

# Initialize results file
> issues_found.jsonl

# Function to search and collect issues
search_and_collect() {
    local query="$1"
    local label="$2"
    
    echo "Searching: $label - '$query'"
    
    # Use gh issue list with search
    gh issue list -R "$REPO" \
        --search "$query" \
        --limit 20 \
        --json number,title,body,labels,state,url,createdAt,closedAt \
        2>/dev/null | jq -c '.[] | . + {search_label: "'$label'"}' >> issues_found.jsonl || true
}

echo ""
echo "========================================"
echo "STEP 1: Searching by Keywords"
echo "========================================"

for keyword in "${KEYWORDS[@]}"; do
    search_and_collect "$keyword" "keyword:$keyword"
    sleep 0.5  # Rate limiting
done

echo ""
echo "========================================"
echo "STEP 2: Searching by Dialect"
echo "========================================"

search_and_collect "label:moore" "dialect:moore"
sleep 0.5

echo ""
echo "========================================"
echo "STEP 3: Searching by Failing Pass"
echo "========================================"

search_and_collect "MooreToCore" "pass:MooreToCore"
sleep 0.5

echo ""
echo "========================================"
echo "STEP 4: Deduplicating Results"
echo "========================================"

if [ -f issues_found.jsonl ]; then
    # Remove duplicates and convert to JSON array
    sort -u issues_found.jsonl | jq -s 'unique_by(.number)' > unique_issues.json
    ISSUE_COUNT=$(jq 'length' unique_issues.json)
    echo "Found $ISSUE_COUNT unique issues"
else
    echo "[]" > unique_issues.json
    echo "No issues found"
fi

echo ""
echo "Done! Results in unique_issues.json"
