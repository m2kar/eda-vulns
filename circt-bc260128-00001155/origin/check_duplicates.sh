#!/bin/bash
set -e

# 提取关键信息
DIALECT=$(jq -r '.dialect // "unknown"' analysis.json)
FAILING_PASS=$(jq -r '.failing_pass // "unknown"' analysis.json)
CRASH_TYPE=$(jq -r '.crash_type // "unknown"' analysis.json)
ASSERTION_MSG=$(jq -r '.assertion_message // ""' analysis.json)
KEYWORDS=$(jq -r '.keywords[]?' analysis.json | head -10)

echo "=== Extracted Information ==="
echo "Dialect: $DIALECT"
echo "Failing Pass: $FAILING_PASS"
echo "Crash Type: $CRASH_TYPE"
echo "Assertion: $ASSERTION_MSG"
echo "Keywords found: $(echo "$KEYWORDS" | wc -l)"
echo ""

# 初始化搜索结果文件
echo "[]" > search_results.json
REPO="llvm/circt"
SEARCH_COUNT=0

# 搜索函数
search_issues() {
    local query="$1"
    local label="$2"
    
    echo "[SEARCH] $label: $query"
    
    result=$(gh issue list -R "$REPO" \
        --search "$query" \
        --state all \
        --limit 20 \
        --json number,title,body,labels,state,url,createdAt \
        2>/dev/null || echo "[]")
    
    if [ "$result" != "[]" ]; then
        echo "$result" >> search_results_raw.jsonl
        count=$(echo "$result" | jq 'length')
        SEARCH_COUNT=$((SEARCH_COUNT + count))
        echo "  → Found $count issues"
    fi
}

# 清理旧文件
rm -f search_results_raw.jsonl unique_issues.json scored_issues.jsonl

# 1. 按关键词搜索（选择关键的几个）
echo "=== Searching by Keywords ==="
search_issues "StringType" "StringType"
search_issues "MooreToCore" "MooreToCore"
search_issues "sanitizeInOut" "sanitizeInOut"
search_issues "type conversion" "type conversion"
search_issues "module port" "module port"

# 2. 按 Dialect 搜索
echo ""
echo "=== Searching by Dialect ==="
search_issues "label:Moore" "Moore dialect"

# 3. 按 Failing Pass 搜索
echo ""
echo "=== Searching by Failing Pass ==="
search_issues "$FAILING_PASS" "MooreToCore pass"

# 4. 按 Assertion 搜索
echo ""
echo "=== Searching by Assertion ==="
search_issues "dyn_cast" "dyn_cast"

# 去重
if [ -f search_results_raw.jsonl ]; then
    cat search_results_raw.jsonl | jq -s 'unique_by(.number)' > unique_issues.json
else
    echo "[]" > unique_issues.json
fi

ISSUE_COUNT=$(jq 'length' unique_issues.json)
echo ""
echo "=== Summary ==="
echo "Total searches executed: 8"
echo "Unique issues found: $ISSUE_COUNT"
