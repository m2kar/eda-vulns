---
name: check-duplicates
description: 检查 CIRCT GitHub Issues 中是否存在重复报告。使用 gh CLI 搜索 llvm/circt 仓库，基于关键词和崩溃签名计算相似度。
argument-hint: (无参数，在当前目录操作)
allowed-tools: Shell(gh:*), Shell(jq:*), Shell(cat:*), Shell(ls:*), Read, Write, Grep
---

# Skill: 重复检查

## 功能描述

使用 GitHub CLI (`gh`) 搜索 llvm/circt 仓库的 Issues，检查是否存在重复报告。基于关键词匹配、assertion message 相似度等多维度计算相似分数。

## 前置条件

- `gh` CLI 已安装
- `gh` CLI 已认证 (`gh auth status`)

## 输入

当前目录必须包含：
- `analysis.json` - 根因分析数据（包含关键词）
- `error.log` - 错误日志（包含 assertion message）

## 输出

- `duplicates.json` - 搜索结果和相似度评分
- `duplicates.md` - 重复检查报告

## 搜索策略

```
┌─────────────────────────────────────────────────────────────────┐
│                    DUPLICATE CHECK WORKFLOW                      │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: Extract Search Terms                                    │
│  ├── Keywords from analysis.json                                 │
│  ├── Assertion message from error.log                            │
│  ├── Dialect and failing pass                                    │
│  └── Crash type                                                  │
├─────────────────────────────────────────────────────────────────┤
│  Step 2: Search GitHub Issues                                    │
│  ├── Search by keywords (open + closed)                          │
│  ├── Search by assertion message                                 │
│  └── Search by dialect label                                     │
├─────────────────────────────────────────────────────────────────┤
│  Step 3: Calculate Similarity Scores                             │
│  ├── Title keyword match (weight: 2.0)                           │
│  ├── Body keyword match (weight: 1.0)                            │
│  ├── Assertion message match (weight: 3.0)                       │
│  └── Label match (weight: 1.5)                                   │
├─────────────────────────────────────────────────────────────────┤
│  Step 4: Rank and Recommend                                      │
│  ├── Sort by similarity score                                    │
│  ├── Determine recommendation                                    │
│  └── Generate report                                             │
└─────────────────────────────────────────────────────────────────┘
```

## 执行步骤

### Step 1: 提取搜索词

```bash
# 检查 gh CLI
if ! command -v gh &> /dev/null; then
    echo "Error: gh CLI not found"
    echo "Install: https://cli.github.com/"
    exit 1
fi

# 检查认证
if ! gh auth status &> /dev/null; then
    echo "Error: gh CLI not authenticated"
    echo "Run: gh auth login"
    exit 1
fi

echo "GitHub CLI ready"

# 从 analysis.json 提取信息
DIALECT=$(jq -r '.dialect // "unknown"' analysis.json 2>/dev/null)
FAILING_PASS=$(jq -r '.failing_pass // "unknown"' analysis.json 2>/dev/null)
CRASH_TYPE=$(jq -r '.crash_type // "unknown"' analysis.json 2>/dev/null)
ASSERTION_MSG=$(jq -r '.assertion_message // ""' analysis.json 2>/dev/null)

# 获取关键词
KEYWORDS=$(jq -r '.keywords[]?' analysis.json 2>/dev/null | head -10)

# 如果 assertion_message 为空，从 error.log 提取
if [ -z "$ASSERTION_MSG" ] && [ -f error.log ]; then
    ASSERTION_MSG=$(grep -E '(Assertion.*failed|LLVM ERROR)' error.log | head -1)
fi

echo "Dialect: $DIALECT"
echo "Failing pass: $FAILING_PASS"
echo "Crash type: $CRASH_TYPE"
echo "Keywords: $KEYWORDS"
echo "Assertion: ${ASSERTION_MSG:0:100}..."
```

### Step 2: 搜索 GitHub Issues

```bash
REPO="llvm/circt"

# 创建临时文件存储搜索结果
> search_results.json
echo "[]" > search_results.json

# 搜索函数
search_issues() {
    local query="$1"
    local search_type="$2"
    
    echo "Searching: $query"
    
    # 使用 gh 搜索 issues
    # 限制返回数量避免过多结果
    gh issue list -R "$REPO" \
        --search "$query" \
        --limit 10 \
        --json number,title,body,labels,state,url,createdAt \
        2>/dev/null || echo "[]"
}

# 1. 按关键词搜索
echo ""
echo "========================================"
echo "Searching by keywords..."
echo "========================================"

for keyword in $KEYWORDS; do
    result=$(search_issues "$keyword" "keyword")
    if [ "$result" != "[]" ]; then
        # 合并结果
        echo "$result" | jq -c '.[]' >> search_results_raw.jsonl
    fi
done

# 2. 按 dialect 搜索
echo ""
echo "Searching by dialect..."
if [ "$DIALECT" != "unknown" ]; then
    result=$(search_issues "label:$DIALECT" "dialect")
    if [ "$result" != "[]" ]; then
        echo "$result" | jq -c '.[]' >> search_results_raw.jsonl
    fi
fi

# 3. 按 failing pass 搜索
echo ""
echo "Searching by failing pass..."
if [ "$FAILING_PASS" != "unknown" ]; then
    result=$(search_issues "$FAILING_PASS" "pass")
    if [ "$result" != "[]" ]; then
        echo "$result" | jq -c '.[]' >> search_results_raw.jsonl
    fi
fi

# 4. 按 assertion 消息搜索（提取关键部分）
echo ""
echo "Searching by assertion..."
if [ -n "$ASSERTION_MSG" ]; then
    # 提取 assertion 的关键部分（去掉文件路径等）
    ASSERTION_KEY=$(echo "$ASSERTION_MSG" | sed 's/.*Assertion/Assertion/' | cut -c1-50)
    result=$(search_issues "\"$ASSERTION_KEY\"" "assertion")
    if [ "$result" != "[]" ]; then
        echo "$result" | jq -c '.[]' >> search_results_raw.jsonl
    fi
fi

# 去重（按 issue number）
if [ -f search_results_raw.jsonl ]; then
    cat search_results_raw.jsonl | jq -s 'unique_by(.number)' > unique_issues.json
else
    echo "[]" > unique_issues.json
fi

ISSUE_COUNT=$(jq 'length' unique_issues.json)
echo ""
echo "Found $ISSUE_COUNT unique issues"
```

### Step 3: 计算相似度分数

```bash
# 相似度计算函数
calculate_similarity() {
    local issue_json="$1"
    local score=0
    
    local title=$(echo "$issue_json" | jq -r '.title')
    local body=$(echo "$issue_json" | jq -r '.body // ""')
    local labels=$(echo "$issue_json" | jq -r '.labels[].name' 2>/dev/null | tr '\n' ' ')
    
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
        ASSERTION_KEY=$(echo "$ASSERTION_MSG" | sed 's/.*Assertion/Assertion/' | cut -c1-30)
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
    
    # 5. Failing pass 匹配 (权重 2.0)
    if [ "$FAILING_PASS" != "unknown" ]; then
        if echo "$title $body" | grep -qi "$FAILING_PASS"; then
            score=$(echo "$score + 2.0" | bc)
        fi
    fi
    
    echo "$score"
}

# 为每个 issue 计算分数
echo ""
echo "========================================"
echo "Calculating similarity scores..."
echo "========================================"

> scored_issues.jsonl

jq -c '.[]' unique_issues.json | while read -r issue; do
    number=$(echo "$issue" | jq -r '.number')
    title=$(echo "$issue" | jq -r '.title')
    
    score=$(calculate_similarity "$issue")
    
    echo "Issue #$number (score: $score): ${title:0:60}..."
    
    # 添加分数到 issue 对象
    echo "$issue" | jq --arg score "$score" '. + {similarity_score: ($score | tonumber)}' >> scored_issues.jsonl
done

# 按分数排序
if [ -f scored_issues.jsonl ]; then
    cat scored_issues.jsonl | jq -s 'sort_by(-.similarity_score)' > sorted_issues.json
else
    echo "[]" > sorted_issues.json
fi
```

### Step 4: 生成推荐

```bash
# 获取最高分数
TOP_SCORE=$(jq '.[0].similarity_score // 0' sorted_issues.json)
TOP_ISSUE=$(jq '.[0]' sorted_issues.json)

echo ""
echo "========================================"
echo "Recommendation"
echo "========================================"

# 推荐逻辑
# - score >= 8.0: review_existing (很可能重复)
# - score >= 4.0: likely_new (相关但可能不同)
# - score < 4.0: new_issue (没有相似 issue)

if (( $(echo "$TOP_SCORE >= 8.0" | bc -l) )); then
    RECOMMENDATION="review_existing"
    CONFIDENCE="high"
    echo "⚠️ HIGH similarity found (score: $TOP_SCORE)"
    echo "   Review existing issues before creating new one"
elif (( $(echo "$TOP_SCORE >= 4.0" | bc -l) )); then
    RECOMMENDATION="likely_new"
    CONFIDENCE="medium"
    echo "📋 MEDIUM similarity found (score: $TOP_SCORE)"
    echo "   Related issues exist but this is likely a new bug"
else
    RECOMMENDATION="new_issue"
    CONFIDENCE="high"
    echo "✅ LOW similarity (score: $TOP_SCORE)"
    echo "   This appears to be a new issue"
fi
```

### 生成输出

#### duplicates.json

```bash
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
    "assertion_message": $(echo "$ASSERTION_MSG" | jq -Rs .)
  },
  "results": {
    "total_found": $ISSUE_COUNT,
    "top_score": $TOP_SCORE,
    "issues": $(cat sorted_issues.json | jq '.[0:5]')
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
EOF

echo ""
echo "duplicates.json created"
```

#### duplicates.md

```bash
cat > duplicates.md << ENDOFMD
# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | $ISSUE_COUNT |
| Top Similarity Score | $TOP_SCORE |
| **Recommendation** | **$RECOMMENDATION** |

## Search Parameters

- **Dialect**: $DIALECT
- **Failing Pass**: $FAILING_PASS
- **Crash Type**: $CRASH_TYPE
- **Keywords**: $(echo $KEYWORDS | tr '\n' ', ')

## Top Similar Issues

ENDOFMD

# 添加 top 5 issues
jq -r '.[0:5] | .[] | "### [#\(.number)](\(.url)) (Score: \(.similarity_score))\n\n**Title**: \(.title)\n\n**State**: \(.state)\n\n**Labels**: \(.labels | map(.name) | join(\", \"))\n\n---\n"' sorted_issues.json >> duplicates.md

cat >> duplicates.md << ENDOFMD

## Recommendation

**Action**: \`$RECOMMENDATION\`

$(case $RECOMMENDATION in
    review_existing) 
        echo "⚠️ **Review Required**"
        echo ""
        echo "A highly similar issue was found. Please review the existing issue(s) before creating a new one."
        echo ""
        echo "**If the existing issue describes the same problem:**"
        echo "- Add your test case as a comment"
        echo "- Update status.json to 'duplicate'"
        echo ""
        echo "**If the issue is different:**"
        echo "- Proceed to generate the bug report"
        echo "- Reference the related issue in your report"
        ;;
    likely_new)
        echo "📋 **Proceed with Caution**"
        echo ""
        echo "Related issues exist but this appears to be a different bug."
        echo ""
        echo "**Recommended:**"
        echo "- Proceed to generate the bug report"
        echo "- Reference related issues in the report"
        echo "- Highlight what makes this bug different"
        ;;
    new_issue)
        echo "✅ **Clear to Proceed**"
        echo ""
        echo "No similar issues were found. This is likely a new bug."
        echo ""
        echo "**Recommended:**"
        echo "- Proceed to generate and submit the bug report"
        ;;
esac)

## Scoring Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion appears in body |
| Dialect label match | 1.5 | If dialect label matches |
| Failing pass match | 2.0 | If failing pass appears in issue |
ENDOFMD

echo "duplicates.md created"
```

## 清理临时文件

```bash
rm -f search_results_raw.jsonl unique_issues.json scored_issues.jsonl sorted_issues.json
echo "Cleaned up temporary files"
```

## 相似度评分参考

| 分数范围 | 含义 | 推荐动作 |
|----------|------|----------|
| >= 8.0 | 高度相似 | 复核现有 Issue |
| 4.0 - 7.9 | 相关 | 继续但引用相关 Issue |
| < 4.0 | 无关 | 创建新 Issue |

## 注意事项

1. **API 限制**：GitHub API 有速率限制，避免过多搜索
2. **认证**：确保 `gh auth login` 已完成
3. **搜索语法**：GitHub 搜索有特定语法，复杂查询可能需要调整
4. **误报**：相似度高不一定是重复，需要人工确认
5. **漏报**：相似度低也可能是重复（描述方式不同），保持警惕
6. **Closed Issues**：也搜索已关闭的 Issue，可能是已修复的 Bug
