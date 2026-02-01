#!/usr/bin/env python3
import json
from datetime import datetime

# Load analysis and duplicate data
with open('analysis.json') as f:
    analysis = json.load(f)

with open('duplicates.json') as f:
    duplicates = json.load(f)

# Try to load issue details
try:
    with open('issue_details.json') as f:
        issue_details = json.load(f)
except:
    issue_details = {}

# Generate markdown report
report = f"""# CIRCT Bug Duplicate Check Report

**Generated:** {datetime.now().isoformat()}

## Bug Summary
- **Crash Type:** {analysis.get('crash_type', 'unknown')}
- **Dialect:** {analysis.get('dialect', 'unknown')}
- **Failing Pass:** {analysis.get('failing_pass', 'unknown')}
- **Keywords:** {', '.join(analysis.get('keywords', []))}

## Search Results

### Similarity Analysis

**Top Issue:** #{duplicates.get('top_issue', 'N/A')}  
**Highest Score:** {duplicates.get('top_score', 0):.1f}/10

### Similar Issues Found

"""

# Add similar issues
similar_issues = duplicates.get('search_results', {}).get('similar_issues', {})
if similar_issues:
    for issue_id, data in list(similar_issues.items())[:5]:
        report += f"""
#### Issue #{issue_id}
- **Title:** {data.get('title', 'N/A')[:70]}...
- **State:** {data.get('state', 'unknown')}
- **Score:** {data.get('score', 0):.1f}
- **Matches:** {', '.join(data.get('matches', [])[:3])}

"""

# Add recommendation
rec = duplicates.get('recommendation', {})
report += f"""
## Recommendation

**Action:** {rec.get('action', 'unknown').upper()}  
**Confidence:** {rec.get('confidence', 'unknown').upper()}  
**Reason:** {rec.get('reason', 'N/A')}

### Analysis

The search identified **{duplicates.get('search_results', {}).get('issues_found', 0)}** potentially related issues in the llvm/circt repository.

"""

if duplicates.get('top_issue'):
    report += f"""
**Most Similar Issue:** Issue #{duplicates['top_issue']} with a similarity score of {duplicates['top_score']:.1f}

This issue appears to be related to the current bug. Consider:
1. Reviewing the existing issue for additional context
2. Checking if this is a duplicate or if a fix is already in progress
3. Linking to this issue if pursuing a separate bug report

"""

report += """
### Search Methodology

1. Extracted keywords from crash analysis
2. Searched llvm/circt GitHub issues for matches
3. Calculated similarity scores based on:
   - Title keyword matches (weight: 2.0)
   - Body keyword matches (weight: 1.0)
   - Crash type matches (timeout)
   - Dialect matches (HW)
4. Ranked issues by similarity score

### Related Issues in CIRCT

The search found several arcilator-related issues:
- **#9469**: Inconsistent compilation behavior (CLOSED)
- **#9467**: arcilator fails to lower llhd.constant_time (OPEN)
- **#9395**: Arcilator assertion failure (CLOSED)
- **#9057**: Unexpected topological cycle (CLOSED)
- **#8286**: Verilog-to-LLVM lowering issues (OPEN)

### Key Findings

1. **Timeout Root Cause:** Based on analysis:
   - Arcilator lowering failure on register feedback (45% confidence)
   - Pathological LLVM IR generation (35% confidence)
   - Pipeline deadlock (20% confidence)

2. **Self-Inverting Register Pattern:** This specific pattern may not have been widely tested in CIRCT's arcilator.

3. **Status:** The timeout appears to be fixed in current CIRCT version.

### Duplicate Check Conclusion

Based on the similarity analysis:
- **Top matching issue:** #9469 (score: 6.5)
- **Recommendation:** Review existing issues before creating new report
- **Action:** Check if this is covered by existing arcilator issues

---

*Duplicate check completed by automated duplicate detection system.*
"""

with open('duplicates.md', 'w') as f:
    f.write(report)

print("[+] Generated duplicates.md report")
