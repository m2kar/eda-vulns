#!/usr/bin/env python3
import json
import subprocess
import sys
import re
from pathlib import Path

# Load analysis
with open('analysis.json') as f:
    analysis = json.load(f)

# Extract search terms
keywords = analysis.get('keywords', [])
dialect = analysis.get('dialect', '')
crash_type = analysis.get('crash_type', '')

print(f"[*] Searching for duplicate issues...")
print(f"[*] Keywords: {keywords}")
print(f"[*] Dialect: {dialect}")
print(f"[*] Crash Type: {crash_type}")

# List of relevant issues to check (from gh search results)
issue_candidates = [
    9467, 9469, 9057, 8865, 8286, 9395, 
    3403, 3936, 8610, 7266, 4269, 4688
]

results = []
issue_details = {}

for issue_num in issue_candidates:
    try:
        # Get issue details
        cmd = f"gh issue view {issue_num} --repo llvm/circt --json title,body,state,labels --jq '.'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            continue
        
        issue_data = json.loads(result.stdout)
        issue_details[issue_num] = issue_data
        
    except Exception as e:
        print(f"[-] Error fetching issue {issue_num}: {e}", file=sys.stderr)
        continue

# Calculate similarity scores
similarity_scores = {}

for issue_num, issue in issue_details.items():
    title = issue.get('title', '').lower()
    body = issue.get('body', '').lower()
    full_text = title + ' ' + body
    labels = [l.get('name', '').lower() for l in issue.get('labels', [])]
    
    score = 0
    matches = []
    
    # Title keyword matches (weight 2.0)
    for kw in keywords:
        if kw.lower() in title:
            score += 2.0
            matches.append(f"title:{kw}")
    
    # Body keyword matches (weight 1.0)
    for kw in keywords:
        if kw.lower() in body and kw.lower() not in title:
            score += 1.0
            matches.append(f"body:{kw}")
    
    # Crash type match
    if crash_type.lower() in full_text:
        score += 2.0
        matches.append(f"crash_type:{crash_type}")
    
    # Dialect match
    if dialect.lower() in full_text or any(d in labels for d in [dialect.lower(), 'arc', 'hw']):
        score += 1.5
        matches.append(f"dialect:{dialect}")
    
    if score > 0:
        similarity_scores[issue_num] = {
            'score': score,
            'matches': matches,
            'title': issue.get('title', ''),
            'state': issue.get('state', ''),
            'labels': labels
        }

# Sort by score
sorted_issues = sorted(similarity_scores.items(), key=lambda x: x[1]['score'], reverse=True)

print(f"\n[*] Found {len(sorted_issues)} potentially related issues:")
for issue_num, data in sorted_issues[:10]:
    print(f"  #{issue_num} (score={data['score']:.1f}): {data['title'][:60]}")

# Determine recommendation
top_score = sorted_issues[0][1]['score'] if sorted_issues else 0
top_issue = sorted_issues[0][0] if sorted_issues else None

if top_score >= 5.0:
    recommendation = "likely_duplicate"
    action = "review_existing"
elif top_score >= 2.0:
    recommendation = "possibly_related"
    action = "review_existing"
else:
    recommendation = "new_issue"
    action = "new_issue"

# Output results
output = {
    "search_results": {
        "total_candidates": len(issue_candidates),
        "issues_found": len(similarity_scores),
        "similar_issues": {str(k): v for k, v in sorted_issues}
    },
    "recommendation": {
        "action": action,
        "confidence": "high" if top_score >= 5.0 else "medium" if top_score >= 2.0 else "low",
        "reason": f"Top score: {top_score:.1f}" + (f" from issue #{top_issue}" if top_issue else "")
    },
    "top_issue": int(top_issue) if top_issue else None,
    "top_score": float(top_score)
}

with open('duplicates.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n[+] Duplicates analysis complete!")
print(f"[+] Top issue: #{top_issue} (score: {top_score:.1f})")
print(f"[+] Recommendation: {recommendation} ({action})")

