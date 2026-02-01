#!/usr/bin/env python3
import json
import subprocess
import sys
from datetime import datetime
from difflib import SequenceMatcher

def run_gh_search(query, limit=20):
    """Search GitHub issues using gh CLI"""
    try:
        result = subprocess.run(
            ["gh", "issue", "list", "-R", "llvm/circt", 
             "--search", query, "--limit", str(limit),
             "--json", "number,title,body,labels,state,url,createdAt"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        return []
    except Exception as e:
        print(f"Error searching: {e}", file=sys.stderr)
        return []

def calculate_similarity(issue, keywords, crash_type, failing_pass):
    """Calculate similarity score for an issue"""
    score = 0.0
    title = issue.get('title', '').lower()
    body = (issue.get('body') or '').lower()
    labels = [label.get('name', '').lower() for label in issue.get('labels', [])]
    combined_text = title + ' ' + body + ' ' + ' '.join(labels)
    
    # Keyword matches (weight 2.0 for title, 1.0 for body/labels)
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in title:
            score += 2.0
        elif keyword_lower in body:
            score += 1.0
    
    # Crash type match (weight 3.0)
    if crash_type.lower() in combined_text:
        score += 3.0
    
    # Failing pass match (weight 2.5)
    if failing_pass.lower() in combined_text:
        score += 2.5
    
    # Specific term matches (weight 1.5 each)
    specific_terms = ['struct', 'type coercion', 'type conversion', 'arcilator', 'moore']
    for term in specific_terms:
        if term.lower() in combined_text:
            score += 1.5
    
    # Label matching (weight 1.0)
    for label in labels:
        if any(kw.lower() in label for kw in keywords):
            score += 1.0
    
    return round(score, 2)

def main():
    # Load analysis.json
    with open('analysis.json', 'r') as f:
        analysis = json.load(f)
    
    keywords = analysis.get('keywords', [])
    crash_type = analysis.get('crash_type', 'timeout')
    failing_pass = analysis.get('failing_pass', 'arcilator')
    
    print("🔍 Starting duplicate issue search...")
    print(f"   Keywords: {', '.join(keywords[:5])}...")
    print(f"   Crash Type: {crash_type}")
    print(f"   Failing Pass: {failing_pass}")
    print()
    
    all_issues = {}
    
    # Define search queries
    searches = [
        ("arcilator timeout", "arcilator + timeout"),
        ("arcilator struct", "arcilator + struct"),
        ("struct type coercion", "struct + type coercion"),
        ("packed struct port", "packed struct + port"),
        ("Moore struct", "Moore + struct"),
        ("HW dialect conversion", "HW + conversion"),
        ("type conversion timeout", "type + timeout"),
        ("implicit conversion hang", "implicit + hang"),
        ("non-terminating struct", "non-terminating + struct"),
    ]
    
    for query, desc in searches:
        print(f"  ⏳ Searching: {desc}...")
        issues = run_gh_search(query, limit=15)
        
        for issue in issues:
            num = issue.get('number')
            if num not in all_issues:
                all_issues[num] = issue
                print(f"    ✓ Found: #{num} - {issue.get('title', '')[:60]}")
    
    print()
    print(f"📊 Found {len(all_issues)} unique issues")
    print()
    
    # Calculate similarity scores
    print("📈 Calculating similarity scores...")
    scored_issues = []
    
    for num, issue in all_issues.items():
        score = calculate_similarity(issue, keywords, crash_type, failing_pass)
        scored_issues.append({
            **issue,
            'similarity_score': score
        })
        print(f"   #{num} (score: {score:5.1f}): {issue.get('title', '')[:55]}...")
    
    # Sort by score
    scored_issues.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    print()
    
    # Determine recommendation
    top_score = scored_issues[0]['similarity_score'] if scored_issues else 0.0
    
    if top_score >= 12.0:
        recommendation = "review_existing"
        confidence = "high"
        reason = "High similarity score indicates likely duplicate"
    elif top_score >= 6.0:
        recommendation = "likely_new"
        confidence = "medium"
        reason = "Related issues found but differences suggest new bug"
    else:
        recommendation = "new_issue"
        confidence = "high"
        reason = "No similar issues found"
    
    # Generate duplicates.json
    duplicates_data = {
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "search_terms": {
            "dialect": analysis.get('dialect', 'unknown'),
            "failing_pass": failing_pass,
            "crash_type": crash_type,
            "timeout_seconds": analysis.get('timeout_seconds', 60),
            "keywords": keywords
        },
        "results": {
            "total_found": len(scored_issues),
            "top_score": top_score,
            "issues": scored_issues[:5] if scored_issues else []
        },
        "recommendation": {
            "action": recommendation,
            "confidence": confidence,
            "reason": reason
        }
    }
    
    with open('duplicates.json', 'w') as f:
        json.dump(duplicates_data, f, indent=2, default=str)
    
    print(f"✓ duplicates.json written")
    
    # Generate duplicates.md
    md_content = f"""# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | {len(scored_issues)} |
| Top Similarity Score | {top_score:.1f} |
| **Recommendation** | **{recommendation.upper()}** |
| Confidence | {confidence.upper()} |

## Search Parameters

- **Dialect**: {analysis.get('dialect', 'unknown')}
- **Failing Pass**: {failing_pass}
- **Crash Type**: {crash_type}
- **Timeout**: {analysis.get('timeout_seconds', 60)}s
- **Keywords**: {', '.join(keywords)}

## Top Similar Issues

"""
    
    if scored_issues:
        for issue in scored_issues[:5]:
            md_content += f"""### [#{issue['number']}]({issue['url']}) - Score: {issue['similarity_score']:.1f}

**Title**: {issue['title']}

**State**: {issue['state']}

**Labels**: {', '.join([l.get('name', '') for l in issue.get('labels', [])])}

**Created**: {issue.get('createdAt', 'unknown')}

---

"""
    else:
        md_content += "No similar issues found.\n\n"
    
    md_content += f"""## Recommendation

**Action**: `{recommendation}`

**Confidence**: {confidence.upper()}

**Reason**: {reason}

"""
    
    if recommendation == "review_existing":
        md_content += """
### 🔴 Review Required

A highly similar issue was found. Please review the existing issue(s) before creating a new one.

**If the existing issue describes the same problem:**
- Add your test case as a comment
- Mark status as 'duplicate'

**If the issue is different:**
- Proceed to generate the bug report
- Reference the related issue in your report
"""
    elif recommendation == "likely_new":
        md_content += """
### 🟡 Proceed with Caution

Related issues exist but this appears to be a different bug.

**Recommended:**
- Proceed to generate the bug report
- Reference related issues in the report
- Highlight what makes this bug different
"""
    else:
        md_content += """
### ✅ Clear to Proceed

No similar issues were found. This is likely a new bug.

**Recommended:**
- Proceed to generate and submit the bug report
"""
    
    with open('duplicates.md', 'w') as f:
        f.write(md_content)
    
    print(f"✓ duplicates.md written")
    print()
    print("=" * 60)
    print(f"RECOMMENDATION: {recommendation.upper()}")
    print(f"Top Score: {top_score:.1f}")
    print(f"Top Issue: #{scored_issues[0]['number'] if scored_issues else 'N/A'}")
    print("=" * 60)

if __name__ == '__main__':
    main()

