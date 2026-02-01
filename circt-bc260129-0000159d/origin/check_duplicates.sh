#!/bin/bash

# 检查 gh CLI
if ! command -v gh &> /dev/null; then
    echo "Error: gh CLI not found"
    exit 1
fi

REPO="llvm/circt"

# 从 analysis.json 提取信息（如果存在）
if [ -f analysis.json ]; then
    DIALECT=$(jq -r '.dialect // "unknown"' analysis.json)
    FAILING_PASS=$(jq -r '.failing_pass // "unknown"' analysis.json)
    CRASH_TYPE=$(jq -r '.crash_type // "unknown"' analysis.json)
    ASSERTION_MSG=$(jq -r '.assertion_message // ""' analysis.json)
    KEYWORDS=$(jq -r '.keywords[]?' analysis.json 2>/dev/null | tr '\n' ' ')
else
    DIALECT="Arc"
    FAILING_PASS="LowerState"
    CRASH_TYPE="assertion"
    ASSERTION_MSG="state type must have a known bit width"
    KEYWORDS="inout arcilator LLHD ref StateType sequential logic"
fi

echo "Dialect: $DIALECT"
echo "Failing pass: $FAILING_PASS"
echo "Keywords: $KEYWORDS"
echo ""

# 搜索函数
search_issues() {
    local query="$1"
    gh issue list -R "$REPO" \
        --search "$query" \
        --limit 10 \
        --json number,title,body,labels,state,url,createdAt 2>/dev/null
}

# 1. 搜索 issue #9574
echo "Checking issue #9574..."
ISSUE_9574=$(gh issue view 9574 -R "$REPO" --json title,body,state,labels 2>/dev/null)
echo "$ISSUE_9574" > issue_9574.json

# 2. 按关键词搜索
echo ""
echo "Searching by keywords..."
RESULTS="[]"

for keyword in inout arcilator LowerState; do
    echo "  Searching for: $keyword"
    result=$(search_issues "$keyword")
    if [ "$result" != "[]" ]; then
        RESULTS=$(echo "$RESULTS" "$result" | jq -s 'unique_by(.number)')
    fi
done

# 输出结果
echo "$RESULTS" | jq '.' > search_results.json
COUNT=$(echo "$RESULTS" | jq 'length')
echo "Found $COUNT unique issues"

# 检查 #9574 是否在搜索结果中
if echo "$RESULTS" | jq -e '.[] | select(.number == 9574)' > /dev/null; then
    echo "Issue #9574 found in search results"
else
    echo "Note: Issue #9574 not in keyword search results"
fi

