#!/usr/bin/env python3
import json
import re
from difflib import SequenceMatcher

# 加载数据
with open('analysis.json', 'r') as f:
    analysis = json.load(f)

with open('search_results.json', 'r') as f:
    search_results = json.load(f)

# 提取我们Bug的关键特征
bug_keywords = analysis.get('keywords', [])
bug_error_msg = analysis.get('error_message', '')
bug_dialect = analysis.get('dialect', '')
bug_tool = analysis.get('tool', '')
bug_pass = analysis.get('pass_name', '')
bug_trigger = analysis.get('trigger_construct', {})

# 构建bug的特征文本
bug_text = ' '.join([
    analysis.get('error_message', ''),
    analysis.get('pass_name', ''),
    analysis.get('tool', ''),
    analysis.get('dialect', ''),
    str(analysis.get('crash_location', {})),
    str(analysis.get('root_cause', {})),
    str(analysis.get('trigger_construct', {}))
]).lower()

print("🔍 Bug 特征文本预览:")
print(f"   关键词: {bug_keywords}")
print(f"   错误信息: {bug_error_msg}")
print(f"   方言: {bug_dialect}, 工具: {bug_tool}, Pass: {bug_pass}")
print()

def calculate_similarity_score(bug_text, issue_title, issue_body):
    """计算相似度分数 (0-100)"""
    issue_text = (issue_title + ' ' + issue_body).lower()
    
    scores = {}
    
    # 1. 关键词匹配 (40%)
    keyword_matches = sum(1 for kw in bug_keywords if kw.lower() in issue_text)
    keyword_score = min(100, (keyword_matches / len(bug_keywords)) * 100) if bug_keywords else 0
    scores['keywords'] = keyword_score
    
    # 2. 错误消息匹配 (30%)
    error_match = bug_error_msg.lower() in issue_text
    error_score = 100 if error_match else 0
    scores['error_message'] = error_score
    
    # 3. Dialect 和 Tool 匹配 (20%)
    tool_match = bug_tool.lower() in issue_text
    dialect_match = bug_dialect.lower() in issue_text
    tool_score = (50 if tool_match else 0) + (50 if dialect_match else 0)
    scores['tool_dialect'] = tool_score
    
    # 4. Pass 名称匹配 (10%)
    pass_match = bug_pass.lower() in issue_text
    pass_score = 100 if pass_match else 0
    scores['pass'] = pass_score
    
    # 5. 序列匹配 (作为辅助参考)
    seq_score = SequenceMatcher(None, bug_text[:500], issue_text[:500]).ratio() * 100
    scores['sequence'] = seq_score
    
    # 加权计算总分
    total_score = (
        keyword_score * 0.40 +
        error_score * 0.30 +
        tool_score * 0.20 +
        pass_score * 0.10 +
        seq_score * 0.00  # 序列作为参考但不计入总分
    )
    
    return round(total_score, 2), scores

# 计算所有Issues的相似度
duplicates = []

for issue in search_results['issues']:
    total_score, detail_scores = calculate_similarity_score(
        bug_text,
        issue['title'],
        issue['body']
    )
    
    duplicates.append({
        'issue_number': issue['number'],
        'title': issue['title'],
        'url': issue['url'],
        'state': issue['state'],
        'similarity_score': total_score,
        'detail_scores': detail_scores,
        'match_details': {
            'keywords_found': [kw for kw in bug_keywords if kw.lower() in (issue['title'] + ' ' + issue['body']).lower()],
            'has_error_message': bug_error_msg.lower() in (issue['title'] + ' ' + issue['body']).lower(),
            'has_tool': bug_tool.lower() in (issue['title'] + ' ' + issue['body']).lower(),
            'has_dialect': bug_dialect.lower() in (issue['title'] + ' ' + issue['body']).lower(),
            'has_pass': bug_pass.lower() in (issue['title'] + ' ' + issue['body']).lower(),
        }
    })

# 按相似度排序
duplicates.sort(key=lambda x: x['similarity_score'], reverse=True)

# 生成报告
report = {
    'analysis_date': '2025-02-01',
    'bug_id': analysis.get('testcase_id', ''),
    'bug_summary': {
        'dialect': bug_dialect,
        'tool': bug_tool,
        'pass': bug_pass,
        'error_message': bug_error_msg,
        'keywords': bug_keywords,
    },
    'search_summary': {
        'queries_used': len(search_results['search_queries']),
        'total_issues_found': len(search_results['issues']),
        'issues_analyzed': len(duplicates),
    },
    'duplicate_check_results': duplicates,
    'recommendation': 'pending',  # 将在下面计算
    'highest_similarity_score': duplicates[0]['similarity_score'] if duplicates else 0,
    'most_similar_issue': duplicates[0]['issue_number'] if duplicates else None,
}

# 确定建议
highest_score = report['highest_similarity_score']
if highest_score >= 80:
    report['recommendation'] = 'likely_duplicate'
    report['recommendation_reason'] = f'找到高度相似的Issue #{duplicates[0]["issue_number"]} (相似度: {highest_score}%)'
elif highest_score >= 50:
    report['recommendation'] = 'review_existing'
    report['recommendation_reason'] = f'找到中等相似度的Issue #{duplicates[0]["issue_number"]} (相似度: {highest_score}%)，需要人工审查'
else:
    report['recommendation'] = 'new_issue'
    report['recommendation_reason'] = f'未找到明显相关的Issue (最高相似度: {highest_score}%)，建议作为新Issue提交'

# 保存JSON结果
with open('duplicates.json', 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("✅ 相似度分析完成")
print(f"\n📊 分析结果:")
print(f"   分析的Bug: {analysis.get('testcase_id', '')}")
print(f"   搜索查询数: {len(search_results['search_queries'])}")
print(f"   找到Issues: {len(duplicates)}")
print(f"   最高相似度: {highest_score}%")
print(f"   最相似Issue: #{report['most_similar_issue']}")
print(f"   建议: {report['recommendation']}")
print()

# 显示详细结果
for i, dup in enumerate(duplicates, 1):
    print(f"\n{i}. Issue #{dup['issue_number']} - 相似度: {dup['similarity_score']}%")
    print(f"   标题: {dup['title']}")
    print(f"   URL: {dup['url']}")
    print(f"   匹配关键词: {dup['match_details']['keywords_found']}")

