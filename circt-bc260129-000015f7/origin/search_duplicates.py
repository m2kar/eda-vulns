#!/usr/bin/env python3
import json
import subprocess
import re
from collections import defaultdict

# Load analysis.json
with open('analysis.json', 'r') as f:
    analysis = json.load(f)

# Extract key information for searching
keywords = {
    'crash_type': analysis.get('crash_type', ''),
    'dialect': analysis.get('dialect', ''),
    'pass': analysis.get('pass', ''),
    'tool': analysis.get('tool', ''),
    'error_message': analysis.get('error_message', ''),
    'assertion_message': analysis.get('assertion_message', ''),
}

print("=" * 60)
print("DUPLICATE CHECK WORKFLOW")
print("=" * 60)
print(f"\nAnalyzing test case: {analysis.get('testcase_id')}")
print(f"Issue Type: {analysis.get('crash_type')} in {analysis.get('dialect')} dialect")
print(f"Pass: {analysis.get('pass')}")
print(f"Tool: {analysis.get('tool')}")
print(f"\nKey Error: {analysis.get('error_message')}")

# Search queries to try
search_queries = [
    f"repo:llvm/circt Mem2Reg class",
    f"repo:llvm/circt Mem2Reg assertion",
    f"repo:llvm/circt LLHD Mem2Reg",
    f"repo:llvm/circt intwidth bitwidth limit",
    f"repo:llvm/circt SystemVerilog class LLHD",
    f"repo:llvm/circt class instantiation Mem2Reg",
]

all_issues = []
seen_issue_numbers = set()

print("\n" + "=" * 60)
print("SEARCHING FOR RELATED ISSUES")
print("=" * 60)

for query in search_queries:
    print(f"\nSearching: {query}")
    try:
        result = subprocess.run(
            ['gh', 'issue', 'list', '--repo', 'llvm/circt', '--limit', '100', '--search', query],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and result.stdout:
            lines = result.stdout.strip().split('\n')
            print(f"  Found {len(lines)} issues")
            for line in lines:
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        issue_num = parts[0].replace('#', '').strip()
                        if issue_num and issue_num not in seen_issue_numbers:
                            seen_issue_numbers.add(issue_num)
                            issue_title = parts[1] if len(parts) > 1 else ""
                            all_issues.append({
                                'number': issue_num,
                                'title': issue_title,
                                'search_query': query
                            })
        else:
            print(f"  No results or error")
    except Exception as e:
        print(f"  Error: {e}")

print(f"\nTotal unique issues found: {len(all_issues)}")

# Fetch details for each issue
print("\n" + "=" * 60)
print("FETCHING ISSUE DETAILS")
print("=" * 60)

detailed_issues = []
for issue in all_issues[:20]:  # Limit to 20 for speed
    issue_num = issue['number']
    print(f"\nFetching issue #{issue_num}...")
    try:
        result = subprocess.run(
            ['gh', 'issue', 'view', issue_num, '--repo', 'llvm/circt', '--json', 
             'number,title,body,labels,state,createdAt'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            issue_data = json.loads(result.stdout)
            detailed_issues.append({
                'number': issue_data.get('number'),
                'title': issue_data.get('title', ''),
                'body': issue_data.get('body', ''),
                'labels': [l.get('name') for l in issue_data.get('labels', [])],
                'state': issue_data.get('state', ''),
                'createdAt': issue_data.get('createdAt', ''),
            })
            print(f"  ✓ Retrieved: {issue_data.get('title', 'N/A')[:60]}...")
    except Exception as e:
        print(f"  Error fetching issue #{issue_num}: {e}")

print(f"\nRetrieved details for {len(detailed_issues)} issues")

# Save raw data for similarity analysis
with open('issues_raw.json', 'w') as f:
    json.dump({
        'search_keywords': keywords,
        'search_queries_used': search_queries,
        'issues_found': detailed_issues,
        'total_found': len(detailed_issues)
    }, f, indent=2)

print("\nRaw data saved to issues_raw.json")
