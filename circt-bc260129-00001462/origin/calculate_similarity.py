#!/usr/bin/env python3
import json
import sys

# Load keywords from analysis.json
with open('analysis.json', 'r') as f:
    analysis = json.load(f)

keywords = analysis.get('keywords', [])
error_msg = analysis.get('error_message', '')
component = analysis.get('failing_component', '')
construct = analysis.get('test_case', {}).get('key_constructs', [])

print(f"Keywords: {keywords}")
print(f"Error msg: {error_msg}")
print(f"Component: {component}")
print(f"Constructs: {construct}")
print("")

# Load issues
with open('unique_issues.json', 'r') as f:
    issues = json.load(f)

def calculate_score(issue, keywords, error_msg, component, construct):
    score = 0
    details = []
    
    title = issue.get('title', '').lower()
    body = issue.get('body', '').lower() if issue.get('body') else ''
    labels = [l.get('name', '').lower() for l in issue.get('labels', [])]
    
    # 1. Keyword overlap (0-10)
    keyword_matches = 0
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in title:
            keyword_matches += 2
        elif kw_lower in body:
            keyword_matches += 1
    
    keyword_score = min(10, keyword_matches)
    score += keyword_score
    if keyword_score > 0:
        details.append(f"keyword_match({keyword_score})")
    
    # 2. Error message match (0-10)
    error_score = 0
    error_fragments = ['not supported yet', 'action block', 'concurrent assert']
    for fragment in error_fragments:
        if fragment in title or fragment in body:
            error_score += 5
    error_score = min(10, error_score)
    score += error_score
    if error_score > 0:
        details.append(f"error_match({error_score})")
    
    # 3. Component match - ImportVerilog (0-10)
    if component.lower() in title or component.lower() in body:
        score += 10
        details.append("component_match(10)")
    
    # 4. Construct match - concurrent assertion (0-10)
    construct_score = 0
    for c in construct:
        if c.lower() in title or c.lower() in body:
            construct_score += 5
    construct_score = min(10, construct_score)
    score += construct_score
    if construct_score > 0:
        details.append(f"construct_match({construct_score})")
    
    # 5. Label match (Moore dialect)
    if 'moore' in labels or 'importverilog' in labels:
        score += 5
        details.append("label_match(5)")
    
    return score, details

# Score all issues
scored = []
for issue in issues:
    score, details = calculate_score(issue, keywords, error_msg, component, construct)
    issue['similarity_score'] = score
    issue['score_details'] = details
    scored.append(issue)

# Sort by score
scored.sort(key=lambda x: x['similarity_score'], reverse=True)

# Save results
with open('scored_issues.json', 'w') as f:
    json.dump(scored, f, indent=2)

# Display top 10
print("=" * 60)
print("TOP SCORING ISSUES")
print("=" * 60)
for i, issue in enumerate(scored[:10], 1):
    print(f"\n{i}. Issue #{issue['number']} (Score: {issue['similarity_score']})")
    print(f"   Title: {issue['title']}")
    print(f"   State: {issue['state']}")
    print(f"   Details: {issue['score_details']}")

