#!/usr/bin/env python3
import json
import subprocess
import sys
from typing import List, Dict, Set

# 从 analysis.json 提取关键词
with open('analysis.json', 'r') as f:
    analysis = json.load(f)

# 关键词提取
dialect = analysis.get('dialect', '')
crash_type = analysis.get('crash_type', '')
component = analysis.get('suspected_component', '')
passes = analysis.get('suspected_passes', [])
root_cause = analysis.get('root_cause_hypothesis', '')

# 构建搜索关键词组合
search_queries = [
    "arcilator timeout",
    "combinational loop always_comb",
    "dynamic array packed struct",
    "ConvertToArcs",
    "infinite loop",
    "unpacked array"
]

print("=== 提取的关键信息 ===")
print(f"Dialect: {dialect}")
print(f"Crash Type: {crash_type}")
print(f"Component: {component}")
print(f"Passes: {passes}")
print(f"\n=== 搜索查询 ===")
for q in search_queries:
    print(f"- {q}")

results = []

# 执行搜索 - 使用正确的 gh search issues 命令
for query in search_queries:
    print(f"\n[搜索] {query}")
    try:
        # 使用 gh search issues 命令
        cmd = f'gh search issues --repo llvm/circt "{query}" --json number,title,state,url --limit 10'
        output = subprocess.check_output(cmd, shell=True, text=True)
        issues = json.loads(output)
        
        for issue in issues:
            results.append({
                'query': query,
                'issue_number': issue['number'],
                'title': issue['title'],
                'state': issue['state'],
                'url': issue['url']
            })
            print(f"  Found #{issue['number']}: {issue['title']} [{issue['state']}]")
    except subprocess.CalledProcessError as e:
        print(f"  Error: {e}")
        continue

# 计算相似度分数
print(f"\n=== 计算相似度分数 ===")

# 关键词权重映射
keyword_weights = {
    'arcilator timeout': 5,
    'combinational loop': 3,
    'dynamic array': 2,
    'packed struct': 2,
    'always_comb': 3,
    'ConvertToArcs': 2,
    'LowerState': 2,
    'SplitLoops': 2,
    'infinite loop': 3,
    'array indexing': 2,
    'unpacked array': 1,
}

# 去重并计算分数
seen_issues = {}
for result in results:
    issue_num = result['issue_number']
    if issue_num not in seen_issues:
        seen_issues[issue_num] = {
            'issue_number': issue_num,
            'title': result['title'],
            'state': result['state'],
            'url': result['url'],
            'matched_keywords': set(),
            'score': 0.0
        }
    
    # 计算这个查询对应的关键词
    query = result['query'].lower()
    title_lower = result['title'].lower()
    combined = f"{query} {title_lower}"
    
    for keyword, weight in keyword_weights.items():
        if keyword.lower() in combined:
            seen_issues[issue_num]['matched_keywords'].add(keyword)
            seen_issues[issue_num]['score'] += weight

# 转换为列表并排序
scored_results = []
for issue in seen_issues.values():
    scored_results.append({
        'issue_number': issue['issue_number'],
        'title': issue['title'],
        'state': issue['state'],
        'score': issue['score'],
        'matched_keywords': sorted(list(issue['matched_keywords'])),
        'url': issue['url']
    })

scored_results.sort(key=lambda x: x['score'], reverse=True)

# 输出得分
for result in scored_results:
    print(f"#{result['issue_number']}: {result['score']:.1f} points - {result['title']}")
    print(f"  Keywords: {', '.join(result['matched_keywords'])}")

# 生成 duplicates.json
duplicates = {
    'search_queries': search_queries,
    'results': scored_results,
    'top_score': scored_results[0]['score'] if scored_results else 0,
    'top_issue': scored_results[0]['issue_number'] if scored_results else None,
}

# 推荐决策
if not scored_results:
    recommendation = 'new_issue'
    reason = 'No similar issues found in llvm/circt repository'
elif scored_results[0]['score'] >= 10:
    recommendation = 'review_existing'
    reason = f"High similarity score {scored_results[0]['score']} with issue #{scored_results[0]['issue_number']}"
elif scored_results[0]['score'] >= 6:
    recommendation = 'likely_new'
    reason = f"Moderate similarity score {scored_results[0]['score']} - review carefully"
else:
    recommendation = 'new_issue'
    reason = f"Low similarity score {scored_results[0]['score']} - likely unique issue"

duplicates['recommendation'] = recommendation
duplicates['reason'] = reason

# 保存 duplicates.json
with open('duplicates.json', 'w') as f:
    json.dump(duplicates, f, indent=2)

print(f"\n=== 推荐结果 ===")
print(f"Recommendation: {recommendation}")
print(f"Reason: {reason}")
print(f"Top Score: {duplicates['top_score']}")
print(f"Top Issue: {duplicates['top_issue']}")

