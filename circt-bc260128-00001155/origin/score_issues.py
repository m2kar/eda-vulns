#!/usr/bin/env python3
import json
import sys

# 提取关键信息
with open('analysis.json') as f:
    analysis = json.load(f)

dialect = analysis.get('dialect', 'unknown')
failing_pass = analysis.get('failing_pass', 'unknown')
assertion_msg = analysis.get('assertion_message', '')
keywords = analysis.get('keywords', [])

# 权重配置
WEIGHTS = {
    'title_keyword': 2.0,
    'body_keyword': 1.0,
    'assertion': 3.0,
    'dialect_label': 1.5,
    'failing_pass': 2.0,
}

def calculate_similarity(issue, keywords, assertion_msg, dialect, failing_pass):
    score = 0.0
    
    title = issue.get('title', '').lower()
    body = issue.get('body', '').lower()
    labels = [l.get('name', '').lower() for l in issue.get('labels', [])]
    labels_str = ' '.join(labels)
    
    # 1. 标题关键词匹配 (权重 2.0)
    for keyword in keywords:
        kw_lower = keyword.lower()
        if kw_lower in title:
            score += WEIGHTS['title_keyword']
    
    # 2. 正文关键词匹配 (权重 1.0)
    for keyword in keywords:
        kw_lower = keyword.lower()
        if kw_lower in body:
            score += WEIGHTS['body_keyword']
    
    # 3. Assertion 消息匹配 (权重 3.0)
    if assertion_msg:
        assertion_key = assertion_msg.lower()[:50]
        if assertion_key in body:
            score += WEIGHTS['assertion']
    
    # 4. Dialect 标签匹配 (权重 1.5)
    if dialect != 'unknown' and dialect.lower() in labels_str:
        score += WEIGHTS['dialect_label']
    
    # 5. Failing pass 匹配 (权重 2.0)
    if failing_pass != 'unknown':
        failing_pass_lower = failing_pass.lower()
        if failing_pass_lower in title or failing_pass_lower in body:
            score += WEIGHTS['failing_pass']
    
    return score

# 读取唯一的 Issues
with open('unique_issues.json') as f:
    issues = json.load(f)

# 计算分数
scored_issues = []
for issue in issues:
    score = calculate_similarity(issue, keywords, assertion_msg, dialect, failing_pass)
    issue['similarity_score'] = score
    scored_issues.append(issue)

# 按分数排序
scored_issues.sort(key=lambda x: x['similarity_score'], reverse=True)

# 保存排序后的结果
with open('sorted_issues.json', 'w') as f:
    json.dump(scored_issues, f, indent=2)

# 打印统计信息
print(f"Total issues scored: {len(scored_issues)}")
print(f"\nTop 5 issues:")
for i, issue in enumerate(scored_issues[:5], 1):
    print(f"{i}. #{issue['number']} (score: {issue['similarity_score']:.1f}): {issue['title'][:60]}...")

if scored_issues:
    top_score = scored_issues[0]['similarity_score']
    print(f"\nTop similarity score: {top_score:.1f}")
