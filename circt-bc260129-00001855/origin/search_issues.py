#!/usr/bin/env python3
import json
import subprocess
import sys
from collections import defaultdict

# 加载分析数据
with open('analysis.json', 'r') as f:
    analysis = json.load(f)

# 提取关键词
keywords = analysis.get('keywords', [])
print(f"📌 提取的关键词: {keywords}\n", file=sys.stderr)

# 搜索策略：使用多个关键词组合搜索
search_queries = [
    f"repo:llvm/circt {keywords[0]} {keywords[1]}",  # arcilator LowerState
    f"repo:llvm/circt StateType llhd.ref",
    f"repo:llvm/circt inout port arc",
    f"repo:llvm/circt arcilator assertion",
    f"repo:llvm/circt LowerStatePass",
    f"repo:llvm/circt llhd.ref type",
]

all_issues = {}
issue_ids = set()

for query in search_queries:
    print(f"🔍 搜索: {query}", file=sys.stderr)
    try:
        result = subprocess.run(
            ["gh", "issue", "list", "--repo", "llvm/circt", "--search", query, "--limit", "10", "--json", "number,title,body,url,state"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            try:
                issues = json.loads(result.stdout)
                print(f"   ✓ 找到 {len(issues)} 个 Issues", file=sys.stderr)
                
                for issue in issues:
                    issue_id = issue['number']
                    if issue_id not in issue_ids:
                        issue_ids.add(issue_id)
                        all_issues[issue_id] = {
                            'number': issue_id,
                            'title': issue['title'],
                            'body': issue.get('body', ''),
                            'url': issue['url'],
                            'state': issue['state']
                        }
            except json.JSONDecodeError:
                print(f"   ⚠ JSON 解析失败", file=sys.stderr)
        else:
            print(f"   ✗ 错误: {result.stderr}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        print(f"   ✗ 超时", file=sys.stderr)
    except Exception as e:
        print(f"   ✗ 异常: {e}", file=sys.stderr)

print(f"\n📊 总共找到 {len(all_issues)} 个唯一 Issues\n", file=sys.stderr)

# 输出结果
results = {
    'search_queries': search_queries,
    'total_issues_found': len(all_issues),
    'issues': list(all_issues.values())
}

with open('search_results.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(json.dumps(results, indent=2, ensure_ascii=False))
