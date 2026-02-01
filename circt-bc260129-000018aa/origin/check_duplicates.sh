#!/bin/bash

# 检查 gh CLI 认证和网络
echo "=== 检查 gh CLI 认证和网络 ==="
if ! gh auth status &>/dev/null; then
    echo "⚠️  gh CLI 未认证或网络不可达"
    OFFLINE_MODE=1
else
    echo "✓ gh CLI 已认证"
    OFFLINE_MODE=0
fi

# 定义搜索查询
declare -a SEARCH_QUERIES=(
    "timeout in:title repo:llvm/circt"
    "nested modules timeout repo:llvm/circt"
    "ConvertToArcs timeout repo:llvm/circt"
    "analyzeFanIn repo:llvm/circt"
    "SplitLoops timeout repo:llvm/circt"
    "always_comb timeout repo:llvm/circt"
    "function call chain moore repo:llvm/circt"
    "MooreToCore timeout repo:llvm/circt"
    "moore dialect timeout repo:llvm/circt"
    "infinite loop nested modules repo:llvm/circt"
)

# 初始化结果数组
declare -A results

echo ""
echo "=== 搜索 llvm/circt Issues ==="

if [ $OFFLINE_MODE -eq 0 ]; then
    # 执行搜索
    for query in "${SEARCH_QUERIES[@]}"; do
        echo "搜索: $query"
        response=$(gh issue list -R llvm/circt -S "$query" --limit 20 --json number,title,labels,createdAt 2>/dev/null || echo "")
        
        if [ -n "$response" ]; then
            echo "$response" | jq -r '.[] | .number' | while read -r issue_num; do
                if [ -n "$issue_num" ]; then
                    echo "  发现 Issue #$issue_num"
                    # 获取详细信息
                    details=$(gh issue view $issue_num -R llvm/circt --json number,title,body,labels,createdAt 2>/dev/null || echo "{}")
                    echo "$details"
                fi
            done
        fi
    done
else
    echo "⚠️  离线模式：无法执行搜索"
fi

