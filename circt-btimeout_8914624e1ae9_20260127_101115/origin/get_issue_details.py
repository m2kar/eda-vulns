#!/usr/bin/env python3
import json
import subprocess

# Get details for top issues
top_issues = [9469, 9467, 9395, 9057, 8286]
issue_details = {}

for issue_num in top_issues:
    cmd = f"gh issue view {issue_num} --repo llvm/circt --json title,body,state,labels,createdAt,closedAt,number"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            issue_details[issue_num] = data
            print(f"[+] Fetched issue #{issue_num}")
        except:
            pass

# Save for report generation
with open('issue_details.json', 'w') as f:
    json.dump(issue_details, f, indent=2)

print(f"[+] Saved details for {len(issue_details)} issues")
