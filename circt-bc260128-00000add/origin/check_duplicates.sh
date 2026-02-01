#!/bin/bash
set -e

REPO="llvm/circt"

# Step 1: 提取搜索词
echo "=== STEP 1: Extract search terms ==="
DIALECT=$(jq -r '.dialect // "unknown"' analysis.json 2>/dev/null)
FAILING_PASS=$(jq -r '.failing_pass // "unknown"' analysis.json 2>/dev/null)
CRASH_TYPE=$(jq -r '.crash_type // "unknown"' analysis.json 2>/dev/null)
ASSERTION_MSG=$(jq -r '.assertion_message // ""' analysis.json 2>/dev/null)
KEYWORDS=$(jq -r '.keywords[]?' analysis.json 2>/dev/null)

echo "Dialect: $DIALECT"
echo "Failing pass: $FAILING_PASS"
echo "Crash type: $CRASH_TYPE"
echo "Assertion: ${ASSERTION_MSG:0:80}..."
echo ""
echo "Keywords:"
echo "$KEYWORDS" | head -10

# Step 2: 搜索 GitHub Issues
echo ""
echo "=== STEP 2: Search GitHub Issues ==="

> search_results_raw.jsonl

search_issues() {
    local query="$1"
    
    gh issue list -R "$REPO" \
        --search "$query" \
        --limit 15 \
        --json number,title,body,labels,state,url \
        2>/dev/null || echo "[]"
}

# 2.1 搜索关键词
echo "Searching by keywords..."
for keyword in $KEYWORDS; do
    result=$(search_issues "$keyword")
    if [ "$result" != "[]" ]; then
        echo "$result" | jq -c '.[]' >> search_results_raw.jsonl 2>/dev/null || true
    fi
done

# 2.2 搜索 dialect
echo "Searching by dialect..."
result=$(search_issues "label:$DIALECT")
if [ "$result" != "[]" ]; then
    echo "$result" | jq -c '.[]' >> search_results_raw.jsonl 2>/dev/null || true
fi

# 2.3 搜索 failing pass
echo "Searching by failing pass..."
PASS_SHORT=$(echo "$FAILING_PASS" | cut -d: -f2 | cut -c1-20)
result=$(search_issues "$PASS_SHORT")
if [ "$result" != "[]" ]; then
    echo "$result" | jq -c '.[]' >> search_results_raw.jsonl 2>/dev/null || true
fi

# 2.4 搜索 assertion 关键部分
echo "Searching by assertion message..."
if [ -n "$ASSERTION_MSG" ]; then
    ASSERTION_KEY=$(echo "$ASSERTION_MSG" | cut -c1-40)
    result=$(search_issues "$ASSERTION_KEY")
    if [ "$result" != "[]" ]; then
        echo "$result" | jq -c '.[]' >> search_results_raw.jsonl 2>/dev/null || true
    fi
fi

# Step 3: 去重
echo ""
echo "=== STEP 3: Deduplicate and prepare for scoring ==="

if [ -f search_results_raw.jsonl ] && [ -s search_results_raw.jsonl ]; then
    # 去重（按 issue number）
    cat search_results_raw.jsonl | jq -s 'unique_by(.number)' > unique_issues.json
else
    echo "[]" > unique_issues.json
fi

ISSUE_COUNT=$(jq 'length' unique_issues.json)
echo "Found $ISSUE_COUNT unique issues"

# Step 4: 计算相似度分数
echo ""
echo "=== STEP 4: Calculate similarity scores ==="

> scored_issues.jsonl

jq -c '.[]' unique_issues.json | while read -r issue; do
    number=$(echo "$issue" | jq -r '.number')
    title=$(echo "$issue" | jq -r '.title')
    body=$(echo "$issue" | jq -r '.body // ""')
    labels=$(echo "$issue" | jq -r '.labels[].name' 2>/dev/null | tr '\n' ' ')
    
    score=0
    
    # 1. 标题关键词匹配 (权重 2.0)
    while IFS= read -r keyword; do
        if [ -n "$keyword" ] && echo "$title" | grep -qi "$keyword"; then
            score=$(echo "$score + 2.0" | bc -l 2>/dev/null || echo $score)
        fi
    done <<< "$KEYWORDS"
    
    # 2. 正文关键词匹配 (权重 1.0)
    while IFS= read -r keyword; do
        if [ -n "$keyword" ] && echo "$body" | grep -qi "$keyword"; then
            score=$(echo "$score + 1.0" | bc -l 2>/dev/null || echo $score)
        fi
    done <<< "$KEYWORDS"
    
    # 3. Assertion 消息匹配 (权重 3.0)
    if [ -n "$ASSERTION_MSG" ]; then
        ASSERTION_KEY=$(echo "$ASSERTION_MSG" | cut -c1-30)
        if echo "$body" | grep -qF "$ASSERTION_KEY"; then
            score=$(echo "$score + 3.0" | bc -l 2>/dev/null || echo $score)
        fi
    fi
    
    # 4. Dialect 标签匹配 (权重 1.5)
    if echo "$labels" | grep -qi "$DIALECT"; then
        score=$(echo "$score + 1.5" | bc -l 2>/dev/null || echo $score)
    fi
    
    # 5. Failing pass 匹配 (权重 2.0)
    if echo "$title $body" | grep -qi "$FAILING_PASS"; then
        score=$(echo "$score + 2.0" | bc -l 2>/dev/null || echo $score)
    fi
    
    echo "Issue #$number (score: $score): ${title:0:60}..."
    
    # 添加分数到 issue 对象
    echo "$issue" | jq --arg score "$score" '. + {similarity_score: ($score | tonumber)}' >> scored_issues.jsonl
done

# 按分数排序
if [ -f scored_issues.jsonl ] && [ -s scored_issues.jsonl ]; then
    cat scored_issues.jsonl | jq -s 'sort_by(-.similarity_score)' > sorted_issues.json
else
    echo "[]" > sorted_issues.json
fi

# Step 5: 获取推荐
echo ""
echo "=== STEP 5: Generate recommendation ==="

TOP_SCORE=$(jq '.[0].similarity_score // 0' sorted_issues.json)
TOP_ISSUE=$(jq '.[0] // {}' sorted_issues.json)
TOP_ISSUE_NUM=$(echo "$TOP_ISSUE" | jq -r '.number // "N/A"')

echo "Top score: $TOP_SCORE"
echo "Top issue: #$TOP_ISSUE_NUM"

# 推荐逻辑
if (( $(echo "$TOP_SCORE >= 8.0" | bc -l 2>/dev/null || echo 0) )); then
    RECOMMENDATION="review_existing"
    CONFIDENCE="high"
    echo "⚠️ HIGH similarity found"
elif (( $(echo "$TOP_SCORE >= 4.0" | bc -l 2>/dev/null || echo 0) )); then
    RECOMMENDATION="likely_new"
    CONFIDENCE="medium"
    echo "📋 MEDIUM similarity found"
else
    RECOMMENDATION="new_issue"
    CONFIDENCE="high"
    echo "✅ LOW similarity - appears to be new"
fi

# Step 6: 生成 duplicates.json
echo ""
echo "=== STEP 6: Generate duplicates.json ==="

TIMESTAMP=$(date -Iseconds)
cat > duplicates.json << EOF
{
  "version": "1.0",
  "timestamp": "$TIMESTAMP",
  "search_terms": {
    "dialect": "$DIALECT",
    "failing_pass": "$FAILING_PASS",
    "crash_type": "$CRASH_TYPE",
    "keywords": $(echo "$KEYWORDS" | jq -R -s 'split("\n") | map(select(length > 0))'),
    "assertion_message": "$ASSERTION_MSG"
  },
  "results": {
    "total_found": $ISSUE_COUNT,
    "top_score": $TOP_SCORE,
    "top_issue_number": "$TOP_ISSUE_NUM",
    "top_5_issues": $(jq '.[0:5]' sorted_issues.json)
  },
  "recommendation": {
    "action": "$RECOMMENDATION",
    "confidence": "$CONFIDENCE",
    "reason": "$(case $RECOMMENDATION in
        review_existing) echo "High similarity score indicates potential duplicate issue" ;;
        likely_new) echo "Related issues found but differences suggest new bug" ;;
        new_issue) echo "No similar issues found in llvm/circt repository" ;;
    esac)"
  }
}
EOF

echo "duplicates.json created"

# Step 7: 生成 duplicates.md
echo ""
echo "=== STEP 7: Generate duplicates.md ==="

cat > duplicates.md << 'MDEOF'
# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | REPLACE_ISSUE_COUNT |
| Top Similarity Score | REPLACE_TOP_SCORE |
| **Recommendation** | **REPLACE_RECOMMENDATION** |
| Confidence | REPLACE_CONFIDENCE |

## Search Parameters

| Parameter | Value |
|-----------|-------|
| Dialect | REPLACE_DIALECT |
| Failing Pass | REPLACE_FAILING_PASS |
| Crash Type | REPLACE_CRASH_TYPE |

## Keywords Used

REPLACE_KEYWORDS

## Top Similar Issues

REPLACE_TOP_ISSUES

## Recommendation Details

**Action**: `REPLACE_RECOMMENDATION`

REPLACE_REASON_DETAIL

## Scoring Methodology

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion appears in body |
| Dialect label match | 1.5 | If dialect label matches |
| Failing pass match | 2.0 | If failing pass appears in issue |

MDEOF

# 替换占位符
sed -i "s/REPLACE_ISSUE_COUNT/$ISSUE_COUNT/g" duplicates.md
sed -i "s/REPLACE_TOP_SCORE/$TOP_SCORE/g" duplicates.md
sed -i "s/REPLACE_RECOMMENDATION/$RECOMMENDATION/g" duplicates.md
sed -i "s/REPLACE_CONFIDENCE/$CONFIDENCE/g" duplicates.md
sed -i "s/REPLACE_DIALECT/$DIALECT/g" duplicates.md
sed -i "s/REPLACE_FAILING_PASS/$FAILING_PASS/g" duplicates.md
sed -i "s/REPLACE_CRASH_TYPE/$CRASH_TYPE/g" duplicates.md

# 添加关键词列表
KEYWORDS_MD=$(echo "$KEYWORDS" | sed 's/^/- /')
sed -i "/## Keywords Used/a\\$KEYWORDS_MD" duplicates.md

# 添加 top 5 issues
TOP_ISSUES=$(jq -r '.[0:5] | .[] | "### [#\(.number)](\(.url)) - Score: \(.similarity_score)\n\n**Title**: \(.title)\n\n**State**: \(.state)\n\n---\n"' sorted_issues.json)
sed -i "s|## Top Similar Issues|## Top Similar Issues\n\n$TOP_ISSUES|g" duplicates.md

# 添加推荐理由
case $RECOMMENDATION in
    review_existing)
        REASON="⚠️ **Review Required**\n\nA highly similar issue was found. Please review the existing issue(s) before creating a new one.\n\n**Recommended Action:**\n- Review the top issue carefully\n- If it's the same bug, add your test case as a comment\n- If different, proceed with generating the bug report"
        ;;
    likely_new)
        REASON="📋 **Proceed with Caution**\n\nRelated issues exist but this appears to be a different bug.\n\n**Recommended Action:**\n- Proceed with generating the bug report\n- Reference related issues in the report\n- Highlight what makes this bug different"
        ;;
    new_issue)
        REASON="✅ **Clear to Proceed**\n\nNo similar issues were found. This is likely a new bug.\n\n**Recommended Action:**\n- Proceed to generate and submit the bug report"
        ;;
esac
sed -i "s|## Recommendation Details|## Recommendation Details\n\n$REASON|g" duplicates.md

echo "duplicates.md created"

# 清理临时文件
rm -f search_results_raw.jsonl unique_issues.json scored_issues.jsonl sorted_issues.json

echo ""
echo "=== DONE ==="
echo "recommendation: $RECOMMENDATION"
echo "top_score: $TOP_SCORE"
echo "top_issue: #$TOP_ISSUE_NUM"

