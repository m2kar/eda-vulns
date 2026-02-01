#!/usr/bin/env python3
import json
from datetime import datetime

# Read duplicates.json
with open('duplicates.json', 'r') as f:
    dups = json.load(f)

# Read analysis.json for context
with open('analysis.json', 'r') as f:
    analysis = json.load(f)

# Generate markdown report
md_content = f"""# CIRCT Duplicate Issue Check Report

**Timestamp**: {dups['search_timestamp']}

## Executive Summary

- **Status**: {dups['recommendation'].upper()}
- **Top Similarity Score**: {dups['top_score']:.1f}
- **Top Related Issue**: #{dups['top_issue']}
- **Total Issues Found**: {dups['total_issues_found']}

## Analyzed Bug Information

- **Test Case ID**: {analysis['testcase_id']}
- **Dialect**: {analysis['dialect']}
- **Crash Type**: {analysis['crash_type']}
- **Root Cause**: {analysis['root_cause']['summary']}

## Search Keywords Used

"""

for i, kw in enumerate(dups['keywords'], 1):
    md_content += f"- {i}. {kw}\n"

md_content += f"""
## Search Results

### Recommendation: {dups['recommendation'].upper()}

"""

if dups['recommendation'] == 'review_existing':
    md_content += f"""
**ACTION**: Review existing GitHub issues, particularly issue #{dups['top_issue']}

The highest similarity score ({dups['top_score']:.1f}) suggests potential duplicates exist.
Please manually verify the top related issues below before filing a new report.

"""
elif dups['recommendation'] == 'likely_new':
    md_content += f"""
**ACTION**: Likely safe to file new issue

While some related issues exist, the similarity scores are below the duplication threshold.
The crash appears to be unique or a variant of existing issues.

"""
else:  # new_issue
    md_content += f"""
**ACTION**: Safe to file new issue

No related issues found in the repository. This appears to be a novel crash.

"""

md_content += f"""
## Top 10 Most Similar Issues

| Rank | Issue | Score | Title |
|------|-------|-------|-------|
"""

for rank, issue in enumerate(dups['issues'][:10], 1):
    title = (issue['title'][:60] + '...') if len(issue['title']) > 60 else issue['title']
    md_content += f"| {rank} | #{issue['number']} | {issue['similarity_score']:.1f} | {title} |\n"

md_content += f"""
## Matched Keywords in Top Issue #{dups['top_issue']}

"""

if dups['top_issue']:
    top_issue = next((i for i in dups['issues'] if i['number'] == dups['top_issue']), None)
    if top_issue and top_issue['matched_keywords']:
        for kw in top_issue['matched_keywords']:
            md_content += f"- {kw}\n"
    else:
        md_content += "No keywords matched for analysis\n"

md_content += f"""
## Details

- **Total issues analyzed**: {dups['total_issues_found']}
- **Duplication threshold**: 10.0
- **Scores >= 10.0**: {len([i for i in dups['issues'] if i['similarity_score'] >= 10])}

## Scoring Methodology

1. Similarity scores based on keyword matching:
   - Direct keyword match: +2 points per keyword
   - Related terms (string, moore, port, type, assertion): +1 point each

2. Score >= 10.0 indicates potential duplicate - manual review recommended

3. GitHub API fallback used if gh CLI search unavailable

---

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Test Case: {analysis['testcase_id']}
"""

# Save markdown
with open('duplicates.md', 'w') as f:
    f.write(md_content)

print("✓ Generated duplicates.md")
print(f"\nSummary:")
print(f"  Recommendation: {dups['recommendation']}")
print(f"  Top Score: {dups['top_score']:.1f}")
print(f"  Top Issue: #{dups['top_issue']}")
print(f"  Issues Analyzed: {dups['total_issues_found']}")
print(f"  Issues with score >= 10.0: {len([i for i in dups['issues'] if i['similarity_score'] >= 10])}")
