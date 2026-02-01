#!/bin/bash

set -e

REPO="llvm/circt"

# 提取关键词
DIALECT=$(jq -r '.dialect' analysis.json)
FAILING_PASS=$(jq -r '.failing_pass' analysis.json)
CRASH_TYPE=$(jq -r '.crash_type' analysis.json)
ASSERTION_MSG=$(jq -r '.assertion_message' analysis.json)
KEYWORDS=$(jq -r '.keywords[]' analysis.json)

echo "======== SEARCH PARAMETERS ========"
echo "Dialect: $DIALECT"
echo "Failing Pass: $FAILING_PASS"
echo "Keywords: $(echo $KEYWORDS | tr '\n' ' ')"
echo ""

echo "======== SEARCHING GITHUB ISSUES ========"

declare -a ALL_NUMBERS
declare -a ALL_TITLES

# 搜索1: MooreToCore
echo "[1/4] Searching: MooreToCore"
RESULT=$(gh issue list -R "$REPO" --search "MooreToCore" --limit 20 --json number,title 2>/dev/null || echo "[]")
while IFS= read -r line; do
    num=$(echo "$line" | jq -r '.number // empty')
    title=$(echo "$line" | jq -r '.title // empty')
    if [ -n "$num" ] && [ -n "$title" ]; then
        ALL_NUMBERS+=("$num")
        ALL_TITLES["$num"]="$title"
    fi
done < <(echo "$RESULT" | jq -c '.[]')

# 搜索2: string port
echo "[2/4] Searching: string port"
RESULT=$(gh issue list -R "$REPO" --search "string port" --limit 20 --json number,title 2>/dev/null || echo "[]")
while IFS= read -r line; do
    num=$(echo "$line" | jq -r '.number // empty')
    title=$(echo "$line" | jq -r '.title // empty')
    if [ -n "$num" ] && [ -n "$title" ]; then
        ALL_NUMBERS+=("$num")
        ALL_TITLES["$num"]="$title"
    fi
done < <(echo "$RESULT" | jq -c '.[]')

# 搜索3: Moore dialect
echo "[3/4] Searching: Moore dialect issues"
RESULT=$(gh issue list -R "$REPO" --search "Moore" --limit 20 --json number,title 2>/dev/null || echo "[]")
while IFS= read -r line; do
    num=$(echo "$line" | jq -r '.number // empty')
    title=$(echo "$line" | jq -r '.title // empty')
    if [ -n "$num" ] && [ -n "$title" ]; then
        ALL_NUMBERS+=("$num")
        ALL_TITLES["$num"]="$title"
    fi
done < <(echo "$RESULT" | jq -c '.[]')

# 搜索4: dyn_cast
echo "[4/4] Searching: dyn_cast errors"
RESULT=$(gh issue list -R "$REPO" --search "dyn_cast" --limit 20 --json number,title 2>/dev/null || echo "[]")
while IFS= read -r line; do
    num=$(echo "$line" | jq -r '.number // empty')
    title=$(echo "$line" | jq -r '.title // empty')
    if [ -n "$num" ] && [ -n "$title" ]; then
        ALL_NUMBERS+=("$num")
        ALL_TITLES["$num"]="$title"
    fi
done < <(echo "$RESULT" | jq -c '.[]')

# 去重
UNIQUE_NUMBERS=($(printf '%s\n' "${ALL_NUMBERS[@]}" | sort -u))
TOTAL_FOUND=${#UNIQUE_NUMBERS[@]}

echo "✓ Found $TOTAL_FOUND unique issues"
echo ""

# 计算相似度
echo "======== TOP ISSUES ========"

BEST_SCORE=0
BEST_ISSUE=""
BEST_TITLE=""

for issue_num in "${UNIQUE_NUMBERS[@]}"; do
    title="${ALL_TITLES[$issue_num]}"
    score=0
    
    # 关键词匹配
    for keyword in $KEYWORDS; do
        if echo "$title" | grep -qi "$keyword"; then
            ((score += 2))
        fi
    done
    
    if [ $score -gt $BEST_SCORE ]; then
        BEST_SCORE=$score
        BEST_ISSUE=$issue_num
        BEST_TITLE=$title
    fi
    
    if [ $score -gt 0 ]; then
        echo "  #$issue_num (score: $score): ${title:0:70}"
    fi
done

echo ""
echo "======== RECOMMENDATION ========"

if [ $BEST_SCORE -ge 8 ]; then
    RECOMMENDATION="review_existing"
    CONFIDENCE="high"
    echo "⚠️ HIGH similarity (score: $BEST_SCORE)"
elif [ $BEST_SCORE -ge 4 ]; then
    RECOMMENDATION="likely_new"
    CONFIDENCE="medium"
    echo "📋 MEDIUM similarity (score: $BEST_SCORE)"
else
    RECOMMENDATION="new_issue"
    CONFIDENCE="high"
    echo "✅ LOW similarity (score: $BEST_SCORE)"
fi

# 生成输出
cat > duplicates.json << JSONEOF
{
  "version": "1.0",
  "timestamp": "$(date -Iseconds)",
  "search_terms": {
    "dialect": "$DIALECT",
    "failing_pass": "$FAILING_PASS",
    "crash_type": "$CRASH_TYPE"
  },
  "results": {
    "total_found": $TOTAL_FOUND,
    "top_score": $BEST_SCORE,
    "top_issue_number": $BEST_ISSUE,
    "top_issue_title": $(echo "$BEST_TITLE" | jq -Rs .)
  },
  "recommendation": {
    "action": "$RECOMMENDATION",
    "confidence": "$CONFIDENCE"
  }
}
JSONEOF

echo ""
echo "✓ duplicates.json created"
jq '.recommendation' duplicates.json

