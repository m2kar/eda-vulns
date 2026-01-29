#!/usr/bin/env python3
"""
CIRCT Issue Submission Script (Sub-Skill 4)

Usage:
    python3 submit_issue.py ./circt-b<id>

Prerequisites:
    - gh CLI authenticated (run `gh auth login` if not)
    - issue.md reviewed and approved
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REPO = 'llvm/circt'


def check_gh_auth():
    try:
        result = subprocess.run(
            ['gh', 'auth', 'status'],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def parse_issue_md(issue_path: Path) -> tuple:
    content = issue_path.read_text()
    
    match = re.match(r'^#\s+(.+?)\n', content)
    title = match.group(1) if match else 'Bug Report'
    
    match = re.search(r'\*\*Labels\*\*:\s*(.+?)(?:\n|$)', content)
    if match:
        labels = [l.strip() for l in match.group(1).split(',')]
    else:
        labels = ['bug']
    
    body = re.sub(r'^#\s+.+?\n', '', content)
    body = re.sub(r'---\n\*\*Labels\*\*:.+$', '', body, flags=re.DOTALL)
    body = body.strip()
    
    return title, body, labels


def create_issue(title: str, body: str, labels: list) -> str:
    label_args = []
    for label in labels:
        label_args.extend(['--label', label])
    
    cmd = [
        'gh', 'issue', 'create',
        '--repo', REPO,
        '--title', title,
        '--body', body,
    ] + label_args
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create issue: {result.stderr}")
    
    match = re.search(r'(https://github\.com/[^\s]+)', result.stdout)
    if match:
        return match.group(1)
    
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description='Submit CIRCT issue to GitHub')
    parser.add_argument('workdir', type=Path, help='Work directory with issue.md')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be submitted')
    args = parser.parse_args()
    
    workdir = args.workdir.resolve()
    
    issue_md = workdir / 'issue.md'
    if not issue_md.exists():
        print(f"❌ Error: {issue_md} not found. Run generate_issue.py first.")
        sys.exit(1)
    
    if not args.dry_run:
        if not check_gh_auth():
            print("❌ Error: GitHub CLI not authenticated. Run `gh auth login`")
            sys.exit(1)
    
    title, body, labels = parse_issue_md(issue_md)
    
    print(f"📋 Issue Preview:")
    print(f"   Title: {title}")
    print(f"   Labels: {', '.join(labels)}")
    print(f"   Body length: {len(body)} chars")
    print()
    
    if args.dry_run:
        print("🔍 Dry run - not submitting")
        print(f"\n--- Body Preview ---\n{body[:500]}...")
        return 0
    
    confirm = input("Submit this issue to llvm/circt? [y/N]: ")
    if confirm.lower() != 'y':
        print("Cancelled")
        return 1
    
    print("🔄 Creating issue...")
    try:
        url = create_issue(title, body, labels)
        print(f"✅ Issue created: {url}")
        
        metadata_path = workdir / 'metadata.json'
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            metadata['submitted'] = {
                'url': url,
                'title': title,
                'labels': labels,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2))
        
        return 0
    
    except RuntimeError as e:
        print(f"❌ {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
