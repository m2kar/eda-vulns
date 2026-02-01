#!/bin/bash

# 检查 gh CLI
if ! command -v gh &> /dev/null; then
    echo "ERROR: gh CLI not found"
    exit 1
fi

if ! gh auth status &> /dev/null; then
    echo "ERROR: gh CLI not authenticated"
    exit 1
fi

echo "✓ GitHub CLI ready"

# 从 analysis.json 提取信息
DIALECT=$(jq -r '.dialect // "unknown"' analysis.json)
FAILING_PASS=$(jq -r '.failing_pass // "unknown"' analysis.json)
CRASH_TYPE=$(jq -r '.crash_type // "unknown"' analysis.json)
ASSERTION_MSG=$(jq -r '.assertion_message // ""' analysis.json)
KEYWORDS=$(jq -r '.keywords[]?' analysis.json | head -10)

echo ""
echo "======== SEARCH PARAMETERS ========"
echo "Dialect: $DIALECT"
echo "Failing Pass: $FAILING_PASS"
echo "Crash Type: $CRASH_TYPE"
echo "Keywords: $(echo $KEYWORDS | tr '\n' ' ')"
echo ""

REPO="llvm/circt"

# Step 1: 搜索关键词
echo "======== SEARCHING GITHUB ISSUES ========"

> search_results_raw.jsonl

search_issues() {
    local query="$1"
    local search_type="$2"
    
    echo "[${search_type}] Searching: $query"
    
    gh issue list -R "$REPO" \
        --search "$query" \
        --limit 15 \
        --json number,title,body,labels,state,url,createdAt \
        2>/dev/null || echo "[]"
}

# 搜索1: 关键词
for keyword in $KEYWORDS; do
    result=$(search_issues "$keyword" "keyword")
    if [ "$result" != "[]" ]; then
        echo "$result" | jq -c '.[]' >> search_results_raw.jsonl
    fi
    sleep 1
done

# 搜索2: dialect 标签
if [ "$DIALECT" != "unknown" ]; then
    result=$(search_issues "label:$DIALECT" "dialect")
    if [ "$result" != "[]" ]; then
        echo "$result" | jq -c '.[]' >> search_results_raw.jsonl
    fi
    sleep 1
fi

# 搜索3: failing pass
if [ "$FAILING_PASS" != "unknown" ]; then
    result=$(search_issues "$FAILING_PASS" "pass")
    if [ "$result" != "[]" ]; then
        echo "$result" | jq -c '.[]' >> search_results_raw.jsonl
    fi
    sleep 1
fi

# 搜索4: assertion message 关键部分
if [ -n "$ASSERTION_MSG" ]; then
    ASSERTION_KEY=$(echo "$ASSERTION_MSG" | sed 's/.*dyn_cast/dyn_cast/' | cut -c1-40)
    result=$(search_issues "$ASSERTION_KEY" "assertion")
    if [ "$result" != "[]" ]; then
        echo "$result" | jq -c '.[]' >> search_results_raw.jsonl
    fi
    sleep 1
fi

# 去重
if [ -f search_results_raw.jsonl ]; then
    cat search_results_raw.jsonl | jq -s 'unique_by(.number) | sort_by(-.createdAt)' > unique_issues.json
else
    echo "[]" > unique_issues.json
fi

ISSUE_COUNT=$(jq 'length' unique_issues.json)
echo ""
echo "✓ Found $ISSUE_COUNT unique issues"

# Step 2: 计算相似度分数
echo ""
echo "======== CALCULATING SCORES ========"

calculate_similarity() {
    local issue_json="$1"
    local score=0
    
    local title=$(echo "$issue_json" | jq -r '.title // ""')
    local body=$(echo "$issue_json" | jq -r '.body // ""')
    local labels=$(echo "$issue_json" | jq -r '.labels[].name // ""' 2>/dev/null | tr '\n' ' ')
    
    # 1. 标题关键词匹配 (权重 2.0)
    for keyword in $KEYWORDS; do
        if echo "$title" | grep -qi "$keyword"; then
            score=$(echo "$score + 2.0" | bc)
        fi
    done
    
    # 2. 正文关键词匹配 (权重 1.0)
    for keyword in $KEYWORDS; do
        if echo "$body" | grep -qi "$keyword"; then
            score=$(echo "$score + 1.0" | bc)
        fi
    done
    
    # 3. Assertion 消息匹配 (权重 3.0)
    if [ -n "$ASSERTION_MSG" ]; then
        ASSERTION_KEY=$(echo "$ASSERTION_MSG" | sed 's/.*dyn_cast/dyn_cast/' | cut -c1-30)
        if echo "$body" | grep -qF "$ASSERTION_KEY"; then
            score=$(echo "$score + 3.0" | bc)
        fi
    fi
    
    # 4. Dialect 标签匹配 (权重 1.5)
    if [ "$DIALECT" != "unknown" ]; then
        if echo "$labels" | grep -qi "$DIALECT"; then
            score=$(echo "$score + 1.5" | bc)
        fi
    fi
    
    echo "$score"
}

> scored_issues.jsonl

jq -c '.[]' unique_issues.json | while read -r issue; do
    number=$(echo "$issue" | jq -r '.number')
    title=$(echo "$issue" | jq -r '.title')
    
    score=$(calculate_similarity "$issue")
    
    echo "  Issue #$number (score: $score): ${title:0:60}..."
    
    echo "$issue" | jq --arg score "$score" '. + {similarity_score: ($score | tonumber)}' >> scored_issues.jsonl
done

# 排序
if [ -f scored_issues.jsonl ]; then
    cat scored_issues.jsonl | jq -s 'sort_by(-.similarity_score)' > sorted_issues.json
else
    echo "[]" > sorted_issues.json
fi

# Step 3: 推荐逻辑
echo ""
echo "======== RECOMMENDATION ========"

TOP_SCORE=$(jq '.[0].similarity_score // 0' sorted_issues.json)
TOP_ISSUE=$(jq '.[0]' sorted_issues.json 2>/dev/null)

if [ "$TOP_SCORE" = "0" ] || [ -z "$TOP_SCORE" ]; then
    RECOMMENDATION="new_issue"
    CONFIDENCE="high"
    echo "✅ NO similar issues found"
    TOP_NUMBER="N/A"
elif (( $(echo "$TOP_SCORE >= 8.0" | bc -l) )); then
    RECOMMENDATION="review_existing"
    CONFIDENCE="high"
    TOP_NUMBER=$(echo "$TOP_ISSUE" | jq -r '.number')
    echo "⚠️ HIGH similarity (score: $TOP_SCORE, Issue #$TOP_NUMBER)"
elif (( $(echo "$TOP_SCORE >= 4.0" | bc -l) )); then
    RECOMMENDATION="likely_new"
    CONFIDENCE="medium"
    TOP_NUMBER=$(echo "$TOP_ISSUE" | jq -r '.number')
    echo "📋 MEDIUM similarity (score: $TOP_SCORE, Issue #$TOP_NUMBER)"
else
    RECOMMENDATION="new_issue"
    CONFIDENCE="high"
    TOP_NUMBER="N/A"
    echo "✅ LOW similarity (score: $TOP_SCORE)"
fi

# Step 4: 生成 duplicates.json
TIMESTAMP=$(date -Iseconds)

cat > duplicates.json << JSONEOF
{
  "version": "1.0",
  "timestamp": "$TIMESTAMP",
  "search_terms": {
    "dialect": "$DIALECT",
    "failing_pass": "$FAILING_PASS",
    "crash_type": "$CRASH_TYPE",
    "keywords": $(echo "$KEYWORDS" | jq -R -s 'split("\n") | map(select(length > 0))'),
    "assertion_message": $(echo "$ASSERTION_MSG" | jq -Rs .)
  },
  "results": {
    "total_found": $ISSUE_COUNT,
    "top_score": $TOP_SCORE,
    "top_issue_number": $(echo "$TOP_ISSUE" | jq -r '.number // null'),
    "top_issue_title": $(echo "$TOP_ISSUE" | jq -r '.title // null')
  },
  "recommendation": {
    "action": "$RECOMMENDATION",
    "confidence": "$CONFIDENCE",
    "reason": "$(case $RECOMMENDATION in
        review_existing) echo "High similarity score indicates potential duplicate" ;;
        likely_new) echo "Related issues found but differences suggest new bug" ;;
        new_issue) echo "No similar issues found" ;;
    esac)"
  }
}
JSONEOF

echo ""
echo "✓ duplicates.json created"
echo ""
echo "======== SUMMARY ========"
jq '.' duplicates.json

# 清理临时文件
rm -f search_results_raw.jsonl unique_issues.json scored_issues.jsonl sorted_issues.json

