#!/usr/bin/env python3
"""
CIRCT Duplicate Issue Checker (Sub-Skill 5)

Searches existing open issues in llvm/circt repository using keywords
from root cause analysis to identify potential duplicates.

Uses gh CLI to search issues.

Usage:
    python3 check_duplicates.py ./circt-b<id>
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


REPO = 'llvm/circt'


def check_gh_available() -> bool:
    try:
        result = subprocess.run(
            ['gh', '--version'],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def search_issues(query: str, label: Optional[str] = None, limit: int = 10) -> list:
    cmd = [
        'gh', 'issue', 'list',
        '--repo', REPO,
        '--state', 'open',
        '--search', query,
        '--limit', str(limit),
        '--json', 'number,title,url,labels,createdAt,body'
    ]
    
    if label:
        cmd.extend(['--label', label])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout)
        return []
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return []


def calculate_similarity(issue: dict, keywords: list, assertion: str, dialect: str) -> float:
    score = 0.0
    
    title = issue.get('title', '').lower()
    body = issue.get('body', '').lower()
    combined = title + ' ' + body
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in title:
            score += 2.0
        elif keyword_lower in body:
            score += 1.0
    
    if assertion:
        assertion_parts = assertion.split()
        for part in assertion_parts:
            if len(part) > 5 and part.lower() in combined:
                score += 3.0
    
    labels = [l.get('name', '').lower() for l in issue.get('labels', [])]
    if dialect and dialect.lower() in labels:
        score += 1.5
    
    if 'assertion' in title.lower() or 'crash' in title.lower():
        score += 0.5
    
    return score


def format_issue_result(issue: dict, similarity: float) -> dict:
    return {
        'number': issue.get('number'),
        'title': issue.get('title'),
        'url': issue.get('url'),
        'labels': [l.get('name') for l in issue.get('labels', [])],
        'created_at': issue.get('createdAt'),
        'similarity_score': round(similarity, 2),
    }


def generate_search_queries(analysis: dict, metadata: dict) -> list:
    queries = []
    
    keywords = analysis.get('keywords', [])
    if keywords:
        queries.append(' '.join(keywords[:3]))
    
    assertion = analysis.get('assertion_context', {}).get('assertion', '')
    if assertion:
        key_parts = []
        for part in assertion.split():
            if len(part) > 5 and not part.startswith('"'):
                key_parts.append(part.strip('()[].,'))
        if key_parts:
            queries.append(' '.join(key_parts[:3]))
    
    dialect = analysis.get('dialect')
    failing_pass = analysis.get('failing_pass')
    
    if dialect:
        queries.append(dialect)
        if failing_pass:
            queries.append(f"{dialect} {failing_pass}")
    
    crash_category = analysis.get('crash_pattern', {}).get('category')
    if crash_category and crash_category != 'Unknown':
        queries.append(crash_category)
    
    test_case = analysis.get('test_case_analysis', {})
    for feature in test_case.get('key_features', [])[:2]:
        queries.append(feature)
    
    seen = set()
    unique_queries = []
    for q in queries:
        q_normalized = q.lower().strip()
        if q_normalized and q_normalized not in seen:
            seen.add(q_normalized)
            unique_queries.append(q)
    
    return unique_queries[:5]


def check_duplicates(workdir: Path) -> dict:
    result = {
        'queries_used': [],
        'potential_duplicates': [],
        'high_similarity': [],
        'recommendation': None,
    }
    
    analysis_path = workdir / 'analysis.json'
    metadata_path = workdir / 'metadata.json'
    
    analysis = {}
    metadata = {}
    
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    
    queries = generate_search_queries(analysis, metadata)
    result['queries_used'] = queries
    
    all_issues = {}
    dialect = analysis.get('dialect')
    
    for query in queries:
        print(f"   Searching: '{query}'")
        
        issues = search_issues(query, label=dialect if dialect else None)
        
        for issue in issues:
            issue_num = issue.get('number')
            if issue_num not in all_issues:
                all_issues[issue_num] = issue
        
        if not issues and dialect:
            issues = search_issues(query)
            for issue in issues:
                issue_num = issue.get('number')
                if issue_num not in all_issues:
                    all_issues[issue_num] = issue
    
    keywords = analysis.get('keywords', [])
    assertion = analysis.get('assertion_context', {}).get('assertion', '')
    
    scored_issues = []
    for issue in all_issues.values():
        similarity = calculate_similarity(issue, keywords, assertion, dialect or '')
        if similarity > 0:
            scored_issues.append((issue, similarity))
    
    scored_issues.sort(key=lambda x: x[1], reverse=True)
    
    for issue, similarity in scored_issues[:10]:
        formatted = format_issue_result(issue, similarity)
        result['potential_duplicates'].append(formatted)
        
        if similarity >= 5.0:
            result['high_similarity'].append(formatted)
    
    if result['high_similarity']:
        result['recommendation'] = 'review_existing'
        result['recommendation_reason'] = (
            f"Found {len(result['high_similarity'])} issue(s) with high similarity. "
            "Review before creating new issue."
        )
    elif result['potential_duplicates']:
        result['recommendation'] = 'likely_new'
        result['recommendation_reason'] = (
            f"Found {len(result['potential_duplicates'])} potentially related issue(s), "
            "but none with high similarity. Likely a new issue."
        )
    else:
        result['recommendation'] = 'new_issue'
        result['recommendation_reason'] = "No similar issues found. Safe to create new issue."
    
    return result


def generate_duplicates_report(check_result: dict) -> str:
    report = []
    report.append("# Duplicate Issue Check Report\n")
    
    report.append("## Recommendation\n")
    rec = check_result.get('recommendation', 'unknown')
    reason = check_result.get('recommendation_reason', '')
    
    icon = {
        'review_existing': '⚠️',
        'likely_new': '✅',
        'new_issue': '✅',
    }.get(rec, '❓')
    
    report.append(f"{icon} **{rec}**")
    report.append(f"\n{reason}")
    report.append("")
    
    high_sim = check_result.get('high_similarity', [])
    if high_sim:
        report.append("## ⚠️ High Similarity Issues\n")
        report.append("These issues are likely related or duplicates:\n")
        for issue in high_sim:
            report.append(f"### #{issue['number']}: {issue['title']}")
            report.append(f"- **URL**: {issue['url']}")
            report.append(f"- **Labels**: {', '.join(issue.get('labels', []))}")
            report.append(f"- **Similarity Score**: {issue['similarity_score']}")
            report.append("")
    
    potential = check_result.get('potential_duplicates', [])
    if potential:
        report.append("## Potentially Related Issues\n")
        for issue in potential:
            if issue not in high_sim:
                report.append(f"- [#{issue['number']}]({issue['url']}): {issue['title']} (score: {issue['similarity_score']})")
        report.append("")
    
    queries = check_result.get('queries_used', [])
    if queries:
        report.append("## Search Queries Used\n")
        for q in queries:
            report.append(f"- `{q}`")
        report.append("")
    
    return '\n'.join(report)


def main():
    parser = argparse.ArgumentParser(description='Check for duplicate CIRCT issues')
    parser.add_argument('workdir', type=Path, help='Work directory')
    parser.add_argument('--skip-if-no-gh', action='store_true',
                        help='Skip silently if gh CLI not available')
    args = parser.parse_args()
    
    workdir = args.workdir.resolve()
    
    if not check_gh_available():
        if args.skip_if_no_gh:
            print("⚠️  gh CLI not available, skipping duplicate check")
            return 0
        print("❌ Error: gh CLI not found. Install GitHub CLI: https://cli.github.com/")
        sys.exit(1)
    
    print(f"🔍 Checking for duplicate issues in {REPO}...")
    
    check_result = check_duplicates(workdir)
    
    duplicates_json = workdir / 'duplicates.json'
    duplicates_json.write_text(json.dumps(check_result, indent=2))
    
    report = generate_duplicates_report(check_result)
    duplicates_md = workdir / 'duplicates.md'
    duplicates_md.write_text(report)
    
    metadata_path = workdir / 'metadata.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata['duplicate_check'] = {
            'recommendation': check_result['recommendation'],
            'high_similarity_count': len(check_result.get('high_similarity', [])),
            'total_related_count': len(check_result.get('potential_duplicates', [])),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))
    
    print(f"\n📊 Duplicate Check Summary:")
    print(f"   Recommendation: {check_result['recommendation']}")
    print(f"   High Similarity Issues: {len(check_result.get('high_similarity', []))}")
    print(f"   Potentially Related: {len(check_result.get('potential_duplicates', []))}")
    
    if check_result.get('high_similarity'):
        print(f"\n   ⚠️  Review these issues before creating new:")
        for issue in check_result['high_similarity'][:3]:
            print(f"      #{issue['number']}: {issue['title'][:60]}...")
    
    print(f"\n📁 Output files:")
    print(f"   {duplicates_json}")
    print(f"   {duplicates_md}")
    print(f"\n   Next step: python3 generate_issue.py {workdir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
