#!/usr/bin/env python3
import json
import re
import sys

# Load issues
with open('unique_issues.json', 'r') as f:
    issues = json.load(f)

# Load analysis
with open('analysis.json', 'r') as f:
    analysis = json.load(f)

# Extract key terms
keywords = set(analysis.get('keywords', []))
dialect = analysis.get('dialect', 'unknown').lower()
failing_pass = analysis.get('failing_pass', 'unknown').lower()
assertion_msg = analysis.get('assertion_message', '').lower()

print(f"Keywords: {keywords}")
print(f"Dialect: {dialect}")
print(f"Failing pass: {failing_pass}")
print(f"Assertion: {assertion_msg[:60]}...")
print()

def calculate_similarity(issue):
    """Calculate similarity score (0-20 scale)"""
    score = 0.0
    title = (issue.get('title', '') or '').lower()
    body = (issue.get('body', '') or '').lower()
    labels = [l.get('name', '').lower() for l in (issue.get('labels', []) or [])]
    
    # 1. Title keyword match (weight: 2.0 per match)
    for kw in keywords:
        if kw.lower() in title:
            score += 2.0
    
    # 2. Body keyword match (weight: 1.0 per match)
    for kw in keywords:
        if kw.lower() in body:
            score += 1.0
    
    # 3. Assertion message match (weight: 3.0)
    if assertion_msg and assertion_msg in body:
        score += 3.0
    
    # 4. Dialect label match (weight: 1.5)
    if any(dialect in l for l in labels):
        score += 1.5
    
    # 5. Failing pass match (weight: 2.0)
    if failing_pass in title or failing_pass in body:
        score += 2.0
    
    # 6. MooreToCore specific (weight: 2.0)
    if 'mooretocore' in title or 'mooretocore' in body:
        score += 2.0
    
    # 7. Type conversion related (weight: 1.5)
    if 'type conversion' in body or 'converttype' in body:
        score += 1.5
    
    # 8. Port/module related (weight: 1.0)
    if 'port' in title or 'port' in body:
        score += 1.0
    
    # 9. String type related (weight: 1.5)
    if 'string' in title or 'stringtype' in body:
        score += 1.5
    
    # 10. Dyn_cast/crash related (weight: 1.0)
    if 'dyn_cast' in body or 'assertion' in body.lower():
        score += 1.0
    
    return score

# Calculate scores
scored_issues = []
for issue in issues:
    score = calculate_similarity(issue)
    issue['similarity_score'] = score
    scored_issues.append(issue)

# Sort by score descending
scored_issues.sort(key=lambda x: x['similarity_score'], reverse=True)

# Save scored issues
with open('scored_issues.json', 'w') as f:
    json.dump(scored_issues, f, indent=2)

# Print top 10
print("=" * 70)
print("TOP SIMILAR ISSUES (Score 0-20)")
print("=" * 70)
print()

for i, issue in enumerate(scored_issues[:10], 1):
    score = issue['similarity_score']
    number = issue['number']
    title = issue['title'][:70]
    url = issue['url']
    state = issue['state']
    
    print(f"{i}. #{number} (Score: {score:.1f}) [{state}]")
    print(f"   {title}")
    print(f"   {url}")
    print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
top_score = scored_issues[0]['similarity_score'] if scored_issues else 0
print(f"Total issues analyzed: {len(scored_issues)}")
print(f"Top similarity score: {top_score:.1f}/20")
print()

if top_score >= 10.0:
    recommendation = "review_existing"
    reason = "High similarity score indicates potential duplicate"
elif top_score >= 5.0:
    recommendation = "likely_new"
    reason = "Related issues found but likely a different bug"
else:
    recommendation = "new_issue"
    reason = "No similar issues found"

print(f"Recommendation: {recommendation}")
print(f"Reason: {reason}")

