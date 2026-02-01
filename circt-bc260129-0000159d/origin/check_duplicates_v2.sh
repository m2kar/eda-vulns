#!/bin/bash

REPO="llvm/circt"

# 检查 issue #9574
echo "Checking issue #9574..."
ISSUE_9574=$(gh issue view 9574 -R "$REPO" --json title,body,state,labels,url 2>/dev/null)
echo "$ISSUE_9574" > issue_9574.json

# 提取 issue #9574 的标题和状态
if [ -n "$ISSUE_9574" ]; then
    ISSUE_TITLE=$(echo "$ISSUE_9574" | jq -r '.title')
    ISSUE_STATE=$(echo "$ISSUE_9574" | jq -r '.state')
    ISSUE_URL=$(echo "$ISSUE_9574" | jq -r '.url')
    echo "  Title: $ISSUE_TITLE"
    echo "  State: $ISSUE_STATE"
    echo "  URL: $ISSUE_URL"
else
    echo "  Issue #9574 not accessible"
fi

# 搜索相关 issues
echo ""
echo "Searching for similar issues..."

SEARCH_TERMS="inout arcilator LowerState arc"

echo "[]" > results.json

for term in $SEARCH_TERMS; do
    echo "  Searching: $term"
    RESULT=$(gh issue list -R "$REPO" --search "$term" --limit 5 --json number,title,state 2>/dev/null)
    if [ "$RESULT" != "[]" ]; then
        # 简单添加到结果
        echo "$RESULT" | jq -c '.[]' >> results_temp.jsonl
    fi
done

# 去重并排序
if [ -f results_temp.jsonl ]; then
    cat results_temp.jsonl | jq -s 'unique_by(.number)' > results.json
    COUNT=$(jq 'length' results.json)
    echo "Found $COUNT unique issues"
    
    # 显示前 5 个
    echo ""
    echo "Top issues:"
    jq -r '.[0:5][] | "#\(.number): \(.title) (\(.state))"' results.json
else
    echo "No results found"
fi

