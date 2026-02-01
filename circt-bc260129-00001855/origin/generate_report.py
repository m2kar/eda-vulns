#!/usr/bin/env python3
import json
from datetime import datetime

# 加载数据
with open('analysis.json', 'r') as f:
    analysis = json.load(f)

with open('duplicates.json', 'r') as f:
    duplicates = json.load(f)

# 生成Markdown报告
report_md = f"""# CIRCT Bug 重复检查报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**分析ID**: {duplicates['bug_id']}

---

## 📋 Bug 摘要

| 项目 | 内容 |
|------|------|
| **Dialect** | {duplicates['bug_summary']['dialect']} |
| **工具** | {duplicates['bug_summary']['tool']} |
| **Pass** | {duplicates['bug_summary']['pass']} |
| **错误信息** | `{duplicates['bug_summary']['error_message']}` |
| **关键词** | {', '.join([f'`{kw}`' for kw in duplicates['bug_summary']['keywords']])} |

---

## 🔍 搜索策略

### 使用的查询

"""

# 添加搜索查询
for i, query in enumerate(analysis.get('keywords', []), 1):
    report_md += f"- Query {i}: `{query}`\n"

report_md += f"""
### 搜索结果统计

- **总查询数**: {duplicates['search_summary']['queries_used']}
- **找到的Issues**: {duplicates['search_summary']['total_issues_found']}
- **分析的Issues**: {duplicates['search_summary']['issues_analyzed']}

---

## 🎯 重复检查结果

### 🚨 建议: **{duplicates['recommendation'].upper()}**

**原因**: {duplicates.get('recommendation_reason', '无')}

### 匹配评分

| Issue # | 相似度 | 标题 | 状态 |
|---------|--------|------|------|
"""

# 添加详细的Issue表格
for item in duplicates['duplicate_check_results']:
    issue_num = item['issue_number']
    score = item['similarity_score']
    title = item['title'][:60] + '...' if len(item['title']) > 60 else item['title']
    state = item['state']
    report_md += f"| #{issue_num} | {score}% | {title} | {state} |\n"

report_md += f"""

---

## 📊 详细分析结果

### 最相似的Issue: #{duplicates['most_similar_issue']}

**相似度**: {duplicates['highest_similarity_score']}%

"""

# 添加最相似Issue的详细信息
if duplicates['duplicate_check_results']:
    top_issue = duplicates['duplicate_check_results'][0]
    report_md += f"""
**标题**: {top_issue['title']}

**URL**: [{top_issue['url']}]({top_issue['url']})

**状态**: {top_issue['state']}

#### 相似度评分详解

"""
    for key, value in top_issue['detail_scores'].items():
        report_md += f"- **{key}**: {value:.1f}%\n"
    
    report_md += f"""
#### 匹配详情

- **匹配的关键词**: {', '.join([f'`{kw}`' for kw in top_issue['match_details']['keywords_found']]) or '无'}
- **错误信息匹配**: {'✅ 是' if top_issue['match_details']['has_error_message'] else '❌ 否'}
- **工具匹配**: {'✅ 是' if top_issue['match_details']['has_tool'] else '❌ 否'}
- **Dialect匹配**: {'✅ 是' if top_issue['match_details']['has_dialect'] else '❌ 否'}
- **Pass匹配**: {'✅ 是' if top_issue['match_details']['has_pass'] else '❌ 否'}

---

### 所有匹配的Issues

"""
    
    for i, issue in enumerate(duplicates['duplicate_check_results'], 1):
        report_md += f"""
#### {i}. Issue #{issue['issue_number']} - 相似度 {issue['similarity_score']}%

**标题**: {issue['title']}

**链接**: {issue['url']}

**状态**: {issue['state']}

**匹配的关键词**:
"""
        if issue['match_details']['keywords_found']:
            for kw in issue['match_details']['keywords_found']:
                report_md += f"- `{kw}`\n"
        else:
            report_md += "- 无\n"

report_md += """
---

## 💡 建议

"""

if duplicates['recommendation'] == 'likely_duplicate':
    report_md += f"""### ⚠️ 可能是重复报告

此Bug与 Issue #{duplicates['most_similar_issue']} 高度相似 (相似度 {duplicates['highest_similarity_score']}%)。

**建议操作**:
1. 审查 Issue #{duplicates['most_similar_issue']} 的内容
2. 如果确认是同一问题，可以关闭此Bug或添加参考链接
3. 如果是不同的问题，请更新Issue描述以明确差异

**参考链接**: https://github.com/llvm/circt/issues/{duplicates['most_similar_issue']}
"""

elif duplicates['recommendation'] == 'review_existing':
    report_md += f"""### 🔍 需要人工审查

此Bug与已有Issues有一定关联性，但相似度处于中等水平 (最高相似度 {duplicates['highest_similarity_score']}%)。

**建议操作**:
1. 仔细审查最相似的Issue: #{duplicates['most_similar_issue']}
2. 比较两个Issues的具体细节和复现步骤
3. 根据差异决定是否为重复或相关问题
4. 如果相关但不完全相同，可以添加交叉引用
"""

else:
    report_md += f"""### ✅ 建议作为新Issue

未找到明显相关的现有Issue (最高相似度仅 {duplicates['highest_similarity_score']}%)。

**建议操作**:
1. 此Bug应该作为新Issue提交到 llvm/circt
2. 确保提供清晰的描述、复现步骤和堆栈跟踪
3. 使用建议的关键词标记Issue
4. 提供最小化的测试用例
"""

report_md += f"""

---

## 📈 搜索查询总结

使用的搜索查询:

"""

# 从search_results获取查询列表 (如果可用)
try:
    with open('search_results.json', 'r') as f:
        search_results = json.load(f)
    for query in search_results['search_queries']:
        report_md += f"- `{query}`\n"
except:
    pass

report_md += f"""

---

## 🔧 技术细节

### Bug 特征

**Pass**: {analysis.get('pass_name', '未知')}

**Dialect**: {analysis.get('dialect', '未知')}

**工具**: {analysis.get('tool', '未知')}

**错误类型**: {analysis.get('crash_type', '未知')}

**关键词**:
"""

for kw in analysis.get('keywords', []):
    report_md += f"- `{kw}`\n"

report_md += f"""

### 根本原因

{analysis.get('root_cause', {}).get('description', '未知')}

**缺失的处理器**: {analysis.get('root_cause', {}).get('missing_handler', '未知')}

**不支持的类型**: {analysis.get('root_cause', {}).get('unsupported_type', '未知')}

### 触发构造

**类型**: {analysis.get('trigger_construct', {}).get('type', '未知')}

**SystemVerilog**: `{analysis.get('trigger_construct', {}).get('systemverilog', '未知')}`

**IR类型**: `{analysis.get('trigger_construct', {}).get('ir_type', '未知')}`

---

## 📝 注意事项

- 相似度分数基于关键词匹配 (40%)、错误信息匹配 (30%)、工具/Dialect匹配 (20%) 和Pass匹配 (10%)
- 搜索结果基于GitHub Issues API的可用数据
- 建议始终进行人工审查以确认重复关系
- 如果Issue已在llvm/circt中存在，可以添加+1反应或新增信息

---

**生成者**: CIRCT Bug 重复检查系统  
**版本**: 1.0  
**最后更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# 保存报告
with open('duplicates.md', 'w') as f:
    f.write(report_md)

print("✅ Markdown报告已生成")
print(f"\n📄 文件: duplicates.md")
print(f"📏 大小: {len(report_md)} 字符")
print("\n" + "="*60)
print(report_md[:1000] + "\n..." if len(report_md) > 1000 else report_md)

