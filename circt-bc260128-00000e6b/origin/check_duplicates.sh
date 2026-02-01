#!/bin/bash
set -e

# Extract search terms from analysis.json
DIALECT=$(jq -r '.dialect // "unknown"' analysis.json 2>/dev/null)
FAILING_PASS=$(jq -r '.failing_pass // "unknown"' analysis.json 2>/dev/null)
CRASH_TYPE=$(jq -r '.crash_type // "unknown"' analysis.json 2>/dev/null)
ASSERTION_MSG=$(jq -r '.assertion_message // ""' analysis.json 2>/dev/null)

# Extract keywords
KEYWORDS=$(jq -r '.keywords[]?' analysis.json 2>/dev/null | head -15)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "DUPLICATE CHECK WORKFLOW"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Dialect: $DIALECT"
echo "Failing Pass: $FAILING_PASS"
echo "Crash Type: $CRASH_TYPE"
echo "Keywords: $(echo "$KEYWORDS" | tr '\n' ', ' | sed 's/,$//')"
echo ""

# Initialize results
REPO="llvm/circt"
> search_results.jsonl

# Search by keywords
echo "Searching by keywords..."
for keyword in $KEYWORDS; do
    result=$(gh issue list -R "$REPO" --search "$keyword" --limit 5 --json number,title,body,labels,state,url,createdAt 2>/dev/null || echo "[]")
    if [ "$result" != "[]" ] && [ -n "$result" ]; then
        echo "$result" | jq -c '.[]' >> search_results.jsonl 2>/dev/null || true
    fi
done

# Search by dialect
echo "Searching by dialect label..."
result=$(gh issue list -R "$REPO" --search "label:$DIALECT" --limit 5 --json number,title,body,labels,state,url,createdAt 2>/dev/null || echo "[]")
if [ "$result" != "[]" ] && [ -n "$result" ]; then
    echo "$result" | jq -c '.[]' >> search_results.jsonl 2>/dev/null || true
fi

# Search by failing pass
echo "Searching by failing pass..."
result=$(gh issue list -R "$REPO" --search "$FAILING_PASS" --limit 5 --json number,title,body,labels,state,url,createdAt 2>/dev/null || echo "[]")
if [ "$result" != "[]" ] && [ -n "$result" ]; then
    echo "$result" | jq -c '.[]' >> search_results.jsonl 2>/dev/null || true
fi

# Deduplicate by issue number
if [ -f search_results.jsonl ] && [ -s search_results.jsonl ]; then
    cat search_results.jsonl | jq -s 'unique_by(.number)' > unique_issues.json
else
    echo "[]" > unique_issues.json
fi

ISSUE_COUNT=$(jq 'length' unique_issues.json)
echo "Found $ISSUE_COUNT unique issues"
echo ""

# Calculate similarity scores
echo "Calculating similarity scores..."
> scored_issues.jsonl

jq -r '.[] | @json' unique_issues.json 2>/dev/null | while read -r issue_json; do
    issue=$(echo "$issue_json" | jq .)
    number=$(echo "$issue" | jq -r '.number')
    title=$(echo "$issue" | jq -r '.title // ""')
    body=$(echo "$issue" | jq -r '.body // ""')
    labels=$(echo "$issue" | jq -r '.labels[].name' 2>/dev/null | tr '\n' ' ')
    
    score=0
    
    # Title keyword match (weight 2.0)
    for keyword in $KEYWORDS; do
        if echo "$title" | grep -qi "$keyword"; then
            score=$((score + 2))
        fi
    done
    
    # Body keyword match (weight 1.0)
    for keyword in $KEYWORDS; do
        if echo "$body" | grep -qi "$keyword"; then
            score=$((score + 1))
        fi
    done
    
    # Assertion message match (weight 3.0)
    if [ -n "$ASSERTION_MSG" ] && echo "$body" | grep -qF "${ASSERTION_MSG:0:40}"; then
        score=$((score + 3))
    fi
    
    # Dialect label match (weight 1.5)
    if [ "$DIALECT" != "unknown" ] && echo "$labels" | grep -qi "$DIALECT"; then
        score=$((score + 1))
    fi
    
    # Failing pass match (weight 2.0)
    if [ "$FAILING_PASS" != "unknown" ] && echo "$title $body" | grep -qi "$FAILING_PASS"; then
        score=$((score + 2))
    fi
    
    echo "$issue" | jq --arg score "$score" '. + {similarity_score: ($score | tonumber)}' >> scored_issues.jsonl
done

# Sort by score
if [ -f scored_issues.jsonl ] && [ -s scored_issues.jsonl ]; then
    cat scored_issues.jsonl | jq -s 'sort_by(-.similarity_score)' > sorted_issues.json
else
    echo "[]" > sorted_issues.json
fi

TOP_SCORE=$(jq '.[0].similarity_score // 0' sorted_issues.json 2>/dev/null)
TOP_ISSUE=$(jq '.[0] | {number, title, url}' sorted_issues.json 2>/dev/null)
TOP_NUMBER=$(echo "$TOP_ISSUE" | jq -r '.number // "N/A"')

echo "Top similarity score: $TOP_SCORE"
echo "Top issue #$TOP_NUMBER"
echo ""

# Determine recommendation
if (( $(echo "$TOP_SCORE >= 8" | bc -l 2>/dev/null || echo "0") )); then
    RECOMMENDATION="review_existing"
    CONFIDENCE="high"
    echo "⚠️  HIGH similarity found (score: $TOP_SCORE)"
elif (( $(echo "$TOP_SCORE >= 4" | bc -l 2>/dev/null || echo "0") )); then
    RECOMMENDATION="likely_new"
    CONFIDENCE="medium"
    echo "📋 MEDIUM similarity found (score: $TOP_SCORE)"
else
    RECOMMENDATION="new_issue"
    CONFIDENCE="high"
    echo "✅ LOW similarity (score: $TOP_SCORE)"
fi

TIMESTAMP=$(date -Iseconds)

# Generate duplicates.json
cat > duplicates.json << EOF
{
  "version": "1.0",
  "timestamp": "$TIMESTAMP",
  "search_terms": {
    "dialect": "$DIALECT",
    "failing_pass": "$FAILING_PASS",
    "crash_type": "$CRASH_TYPE",
    "keywords": $(echo "$KEYWORDS" | jq -Rs 'split("\n") | map(select(length > 0))'),
    "assertion_message": $(echo "$ASSERTION_MSG" | jq -Rs .)
  },
  "results": {
    "total_found": $ISSUE_COUNT,
    "top_score": $TOP_SCORE,
    "top_issue": $(jq '.[0]' sorted_issues.json 2>/dev/null || echo 'null')
  },
  "recommendation": {
    "action": "$RECOMMENDATION",
    "confidence": "$CONFIDENCE",
    "reason": "$(case $RECOMMENDATION in
        review_existing) echo "High similarity score indicates potential duplicate" ;;
        likely_new) echo "Related issues exist but differences suggest new bug" ;;
        new_issue) echo "No similar issues found" ;;
    esac)"
  }
}
EOF

# Generate duplicates.md
cat > duplicates.md << 'MDEOF'
# Duplicate Check Report

## Summary
MDEOF

cat >> duplicates.md << EOF

| Metric | Value |
|--------|-------|
| Issues Found | $ISSUE_COUNT |
| Top Similarity Score | $TOP_SCORE |
| **Recommendation** | **$RECOMMENDATION** |

## Search Parameters

- **Dialect**: $DIALECT
- **Failing Pass**: $FAILING_PASS
- **Crash Type**: $CRASH_TYPE
- **Keywords**: $(echo "$KEYWORDS" | tr '\n' ',' | sed 's/,$//')

## Top Similar Issues

EOF

jq -r '.[0:5] | .[] | "### [#\(.number)](\(.url)) - Score: \(.similarity_score)\n\n**Title**: \(.title)\n\n**State**: \(.state)\n\n---\n"' sorted_issues.json >> duplicates.md 2>/dev/null || echo "No issues found" >> duplicates.md

cat >> duplicates.md << EOF

## Recommendation

**Action**: \`$RECOMMENDATION\`

EOF

if [ "$RECOMMENDATION" = "review_existing" ]; then
    cat >> duplicates.md << 'EOF'

### ⚠️ Review Required

A highly similar issue was found. Please review the existing issue(s) before creating a new one.

**If the existing issue describes the same problem:**
- Add your test case as a comment
- Reference the original analysis in the comment
- Mark status as 'duplicate'

**If the issue is different:**
- Note the differences from existing issues
- Include reference to related issues in your report
- Proceed with bug report generation

EOF
elif [ "$RECOMMENDATION" = "likely_new" ]; then
    cat >> duplicates.md << 'EOF'

### 📋 Proceed with Caution

Related issues exist but this appears to be a different bug.

**Recommended actions:**
- Review the related issues carefully
- Highlight what makes this bug unique
- Include references to related issues in your report
- Proceed to generate the bug report

EOF
else
    cat >> duplicates.md << 'EOF'

### ✅ Clear to Proceed

No similar issues were found. This is likely a new bug.

**Recommended actions:**
- Proceed to generate and submit the bug report
- Include the test case and root cause analysis

EOF
fi

cat >> duplicates.md << 'EOF'

## Scoring Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2 | Per keyword found in title |
| Body keyword match | 1 | Per keyword found in body |
| Assertion message match | 3 | If assertion matches |
| Dialect label match | 1 | If dialect label matches |
| Failing pass match | 2 | If failing pass appears in issue |

---
Generated: $(date)
EOF

# Cleanup
rm -f search_results.jsonl unique_issues.json scored_issues.jsonl sorted_issues.json search_results_raw.jsonl

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ DUPLICATE CHECK COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Recommendation: $RECOMMENDATION"
echo "Top Score: $TOP_SCORE"
echo "Top Issue: #$TOP_NUMBER"
echo ""
echo "Output files:"
echo "  - duplicates.json (structured results)"
echo "  - duplicates.md (human-readable report)"

