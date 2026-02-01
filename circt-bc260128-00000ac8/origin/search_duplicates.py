#!/usr/bin/env python3
import json
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Tuple

def run_gh_search(query: str) -> List[Dict]:
    """执行 gh issue search，返回解析的结果"""
    try:
        result = subprocess.run(
            ["gh", "issue", "list", "-R", "llvm/circt", 
             "--search", query, "--limit", "30", "--json", "number,title,body,labels,state"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            print(f"❌ Search failed: {result.stderr}", file=sys.stderr)
            return []
    except Exception as e:
        print(f"❌ Error during search: {e}", file=sys.stderr)
        return []

def extract_keywords():
    """从 analysis.json 提取关键词"""
    with open("analysis.json") as f:
        analysis = json.load(f)
    
    keywords = {
        "tool": analysis.get("tool", ""),  # arcilator
        "pass": analysis.get("pass", ""),  # InferStateProperties
        "crash_type": analysis.get("crash_type", ""),  # assertion
        "dialect": analysis.get("dialect", ""),  # arc
        "function": analysis["crash_location"].get("function", ""),  # applyEnableTransformation
        "cast_error": "cast<IntegerType>",  # 从断言消息提取
        "struct_type": "packed struct",  # 从实际类型提取
        "struct_array": "struct array",
        "file": analysis["crash_location"].get("file", "").split("/")[-1],
    }
    return keywords

def calculate_similarity(issue: Dict, keywords: Dict, search_query: str) -> float:
    """计算相似度分数 (0-20)"""
    score = 0.0
    title = issue.get("title", "").lower()
    body = issue.get("body", "").lower()
    combined = f"{title} {body}"
    
    # 完全匹配 (20分): 相同函数和错误类型
    if (keywords["function"] in combined or 
        keywords["pass"] in combined) and keywords["crash_type"] in combined:
        score += 20.0
        return score
    
    # 高度相关 (15分): 同一 pass，不同错误
    if keywords["pass"] in combined:
        if keywords["dialect"] in combined:
            score += 15.0
            return score
        score += 12.0
    
    # 中度相关 (10分): 同一 dialect
    if keywords["dialect"] in combined:
        if "struct" in combined or "cast" in combined:
            score += 10.0
            return score
        score += 7.0
    
    # 弱相关 (5分): 相同错误类型或工具
    if keywords["tool"] in combined or keywords["crash_type"] in combined:
        score += 5.0
    
    if "packed struct" in combined or "struct array" in combined:
        score += 3.0
    
    if "IntegerType" in combined or "cast<" in combined:
        score += 2.0
    
    return min(score, 20.0)

def main():
    print("🔍 Extracting keywords from analysis.json...", file=sys.stderr)
    keywords = extract_keywords()
    
    print(f"Keywords: {keywords}", file=sys.stderr)
    
    # 定义搜索查询
    searches = [
        "arcilator crash",
        "InferStateProperties assertion",
        "packed struct",
        "cast<IntegerType>",
        "struct array",
    ]
    
    all_results = {}
    issue_set = set()  # 去重
    
    print(f"🔎 Searching {len(searches)} queries...", file=sys.stderr)
    
    for query in searches:
        print(f"  • Searching: {query}", file=sys.stderr)
        results = run_gh_search(query)
        all_results[query] = results
        
        for issue in results:
            issue_num = issue["number"]
            if issue_num not in issue_set:
                issue_set.add(issue_num)
    
    # 计算相似度并排序
    scored_issues = []
    for query, issues in all_results.items():
        for issue in issues:
            # 避免重复计分
            existing = next((x for x in scored_issues if x["number"] == issue["number"]), None)
            similarity = calculate_similarity(issue, keywords, query)
            
            if existing:
                # 保留最高分
                existing["similarity"] = max(existing["similarity"], similarity)
                existing["queries"].append(query)
            else:
                scored_issues.append({
                    "number": issue["number"],
                    "title": issue.get("title", ""),
                    "state": issue.get("state", ""),
                    "labels": issue.get("labels", []),
                    "similarity": similarity,
                    "queries": [query]
                })
    
    # 排序
    scored_issues.sort(key=lambda x: -x["similarity"])
    
    # 生成输出
    output = {
        "timestamp": datetime.now().isoformat(),
        "keywords": keywords,
        "total_results": len(scored_issues),
        "top_5": scored_issues[:5],
        "all_results": scored_issues,
        "search_queries": searches
    }
    
    with open("duplicates.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
