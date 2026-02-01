#!/usr/bin/env python3
import json
from datetime import datetime

# Load scored issues
with open('scored_issues.json', 'r') as f:
    scored_issues = json.load(f)

# Load analysis for context
with open('analysis.json', 'r') as f:
    analysis = json.load(f)

# Get top issue
top_issue = scored_issues[0] if scored_issues else None
top_score = top_issue['similarity_score'] if top_issue else 0

# Determine recommendation
if top_score >= 25:
    recommendation = "review_existing"
    confidence = "high"
    reason = "Strong similarity found - likely a duplicate or related feature request"
elif top_score >= 15:
    recommendation = "likely_new"
    confidence = "medium"
    reason = "Related issues exist but appear to address different aspects"
else:
    recommendation = "new_issue"
    confidence = "high"
    reason = "No highly similar issues found"

# Generate duplicates.json
duplicates_json = {
    "version": "1.0",
    "timestamp": datetime.now().isoformat(),
    "search_context": {
        "keywords": analysis.get('keywords', []),
        "error_message": analysis.get('error_message', ''),
        "failing_component": analysis.get('failing_component', ''),
        "dialect": analysis.get('dialect', ''),
        "crash_type": analysis.get('crash_type', '')
    },
    "search_results": {
        "total_issues_found": len(scored_issues),
        "top_10_issues": [
            {
                "number": issue['number'],
                "title": issue['title'],
                "state": issue['state'],
                "url": issue['url'],
                "similarity_score": issue['similarity_score'],
                "score_breakdown": issue.get('score_details', []),
                "created_at": issue.get('createdAt', 'N/A')
            }
            for issue in scored_issues[:10]
        ]
    },
    "top_issue": {
        "number": top_issue['number'],
        "title": top_issue['title'],
        "state": top_issue['state'],
        "url": top_issue['url'],
        "similarity_score": top_score,
        "score_breakdown": top_issue.get('score_details', []),
        "created_at": top_issue.get('createdAt', 'N/A')
    } if top_issue else None,
    "assessment": {
        "recommendation": recommendation,
        "confidence": confidence,
        "reason": reason,
        "top_score": top_score,
        "score_scale": "0-40 (keyword_overlap:10 + error_match:10 + component_match:10 + construct_match:10)"
    }
}

# Generate duplicates.md
markdown = f"""# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Total Issues Found | {len(scored_issues)} |
| Top Similarity Score | {top_score} / 40 |
| **Recommendation** | **{recommendation.upper()}** |
| Confidence | {confidence.upper()} |

## Search Context

**Error**: {analysis.get('error_message', 'N/A')}

**Keywords Searched**:
{chr(10).join([f"- {kw}" for kw in analysis.get('keywords', [])])}

**Component**: {analysis.get('failing_component', 'N/A')}

**Dialect**: {analysis.get('dialect', 'N/A')}

**Crash Type**: {analysis.get('crash_type', 'N/A')}

## Top Issue

"""

if top_issue:
    markdown += f"""### Issue #{top_issue['number']}: {top_issue['title']}

- **State**: {top_issue['state']}
- **Similarity Score**: {top_score} / 40
- **Score Breakdown**: {', '.join(top_issue.get('score_details', []))}
- **Created**: {top_issue.get('createdAt', 'N/A')}
- **URL**: {top_issue['url']}

**Relevance**: This issue has the highest similarity score. Reviewing it is recommended before creating a new report.

"""
else:
    markdown += "No issues found.\n"

# Top 5 other issues
markdown += "## Top 5 Other Related Issues\n\n"
for i, issue in enumerate(scored_issues[1:6], 1):
    markdown += f"{i}. **[#{issue['number']}]({issue['url']})**: {issue['title']}\n"
    markdown += f"   - Score: {issue['similarity_score']}/40\n"
    markdown += f"   - State: {issue['state']}\n\n"

markdown += f"""## Assessment

**Recommendation**: `{recommendation}`

"""

if recommendation == "review_existing":
    markdown += """**Action Required**: 

A highly similar issue was found. Before creating a new issue:
1. Review the existing issue(s) to understand the scope
2. Check if your test case is already covered
3. If the same issue, comment with additional context/test cases
4. If different, create a new issue referencing the related one

"""
elif recommendation == "likely_new":
    markdown += """**Action Required**:

Related issues exist but address different aspects or components. You can proceed with creating a new issue, but:
1. Reference the related issues in your report
2. Clearly explain what makes this issue different
3. Provide your specific test case that demonstrates the limitation

"""
else:
    markdown += """**Action Required**:

No highly similar issues were found. You are clear to proceed with creating a comprehensive bug report.

"""

markdown += """## Scoring Methodology

| Factor | Max Points | Description |
|--------|-----------|-------------|
| Keyword Match | 10 | Presence of keywords in title/body |
| Error Message Match | 10 | Presence of error-specific terms |
| Component Match | 10 | Matching ImportVerilog component |
| Construct Match | 10 | Matching concurrent assertion constructs |
| **Total Scale** | **40** | Sum of all factors |

---

Generated: {timestamp}
""".format(timestamp=datetime.now().isoformat())

# Save JSON
with open('duplicates.json', 'w') as f:
    json.dump(duplicates_json, f, indent=2)

# Save Markdown
with open('duplicates.md', 'w') as f:
    f.write(markdown)

print("✓ duplicates.json generated")
print("✓ duplicates.md generated")
print("")
print("=" * 60)
print("FINAL RECOMMENDATION")
print("=" * 60)
print(f"Recommendation: {recommendation.upper()}")
print(f"Confidence: {confidence.upper()}")
print(f"Top Score: {top_score} / 40")
if top_issue:
    print(f"Top Issue: #{top_issue['number']} - {top_issue['title']}")
print(f"Reason: {reason}")

