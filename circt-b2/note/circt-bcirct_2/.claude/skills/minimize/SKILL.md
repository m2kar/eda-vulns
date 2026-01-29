---
name: minimize
description: 最小化 CIRCT 崩溃测例。基于根因分析保留关键构造，迭代删除无关代码，每步验证崩溃可复现。输出最小化测例和复现命令。
argument-hint: (无参数，在当前目录操作)
allowed-tools: Shell(circt-verilog:*), Shell(firtool:*), Shell(circt-opt:*), Shell(cat:*), Shell(ls:*), Shell(wc:*), Shell(diff:*), Shell(cp:*), Shell(mv:*), Shell(rm:*), Read, Write, Grep
---

# Skill: 测例最小化

## 功能描述

对崩溃测例进行最小化处理，在保证崩溃可复现的前提下，删除所有无关代码。使用根因分析数据（`analysis.json`）指导保留关键构造，避免过度删除。

## 输入

当前目录必须包含：
- `source.sv` (或 `.fir`/`.mlir`) - 原始测例
- `error.txt` - 原始错误日志
- `metadata.json` - 复现元数据
- `analysis.json` - 根因分析数据（可选，但推荐）

## 输出

- `bug.sv` - 最小化测例
- `error.log` - 最小化后的错误日志
- `command.txt` - 单行复现命令
- `minimize_report.md` - 最小化过程报告

## 最小化原则

### 保守策略

```
宁多勿少原则：
- 不确定是否需要时，保留
- 关键构造必须保留
- 每步验证崩溃可复现
- 保持代码语法正确
```

### 关键构造来源

从 `analysis.json` 提取：

```bash
# 获取关键构造
KEY_CONSTRUCTS=$(jq -r '.test_case.key_constructs[]?' analysis.json 2>/dev/null)

# 获取问题模式
PROBLEMATIC_PATTERNS=$(jq -r '.test_case.problematic_patterns[]?' analysis.json 2>/dev/null)

# 获取关键词（用于识别关键代码行）
KEYWORDS=$(jq -r '.keywords[]?' analysis.json 2>/dev/null)
```

## 最小化流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    MINIMIZATION WORKFLOW                         │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: Preparation                                            │
│  ├── Copy source to working file                                 │
│  ├── Extract reproduction command                                │
│  ├── Verify original crash                                       │
│  └── Extract assertion signature                                 │
├─────────────────────────────────────────────────────────────────┤
│  Phase 2: Module-level Reduction                                 │
│  ├── Identify standalone modules                                 │
│  ├── Try removing each non-essential module                      │
│  └── Verify crash after each removal                             │
├─────────────────────────────────────────────────────────────────┤
│  Phase 3: Statement-level Reduction                              │
│  ├── Identify removable statements                               │
│  ├── Skip key constructs (from analysis.json)                    │
│  ├── Try removing each statement                                 │
│  └── Verify crash after each removal                             │
├─────────────────────────────────────────────────────────────────┤
│  Phase 4: Cleanup                                                │
│  ├── Remove comments                                             │
│  ├── Remove empty lines                                          │
│  ├── Remove unused variables/ports                               │
│  └── Final verification                                          │
├─────────────────────────────────────────────────────────────────┤
│  Phase 5: Validation                                             │
│  ├── Verify final crash matches original                         │
│  ├── Generate command.txt                                        │
│  └── Generate minimize_report.md                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 执行步骤

### Phase 1: 准备

```bash
# 检测源文件类型
SOURCE_FILE=""
for ext in sv fir mlir; do
    if [ -f source.$ext ]; then
        SOURCE_FILE="source.$ext"
        SOURCE_EXT="$ext"
        break
    fi
done

if [ -z "$SOURCE_FILE" ]; then
    echo "Error: No source file found"
    exit 1
fi

# 获取原始行数
ORIGINAL_LINES=$(wc -l < "$SOURCE_FILE")
echo "Original source: $SOURCE_FILE ($ORIGINAL_LINES lines)"

# 复制到工作文件
cp "$SOURCE_FILE" working.$SOURCE_EXT

# 获取复现命令
REPRO_CMD=$(jq -r '.reproduction.command' metadata.json)
echo "Reproduction command: $REPRO_CMD"

# 提取 assertion 签名
ORIGINAL_ASSERTION=$(grep -E '(Assertion.*failed|LLVM ERROR|error:|fatal:)' error.txt | head -1)
echo "Original assertion: $ORIGINAL_ASSERTION"
```

### Phase 2: 模块级删除

对于 SystemVerilog，尝试删除非必要模块：

```bash
# 识别所有模块
MODULES=$(grep -E '^(module|interface|package|class)\s+\w+' working.sv | awk '{print $2}' | sed 's/[;(].*//')

echo "Found modules: $MODULES"

# 尝试删除每个模块（保留触发崩溃的主模块）
for MODULE in $MODULES; do
    echo "Trying to remove module: $MODULE"
    
    # 创建备份
    cp working.sv working.sv.bak
    
    # 尝试删除模块（简单策略：删除从 "module X" 到 "endmodule" 的所有内容）
    # 注意：这是简化的示例，实际实现需要更精确的解析
    
    # 运行验证
    eval $REPRO_CMD > test_output.log 2>&1
    
    # 检查是否仍然崩溃
    if grep -qE '(Assertion.*failed|LLVM ERROR)' test_output.log; then
        NEW_ASSERTION=$(grep -E '(Assertion.*failed|LLVM ERROR)' test_output.log | head -1)
        if [ "$NEW_ASSERTION" = "$ORIGINAL_ASSERTION" ]; then
            echo "  ✓ Module removed, crash preserved"
        else
            echo "  ✗ Different crash, restoring"
            cp working.sv.bak working.sv
        fi
    else
        echo "  ✗ No crash, restoring"
        cp working.sv.bak working.sv
    fi
done
```

### Phase 3: 语句级删除

```bash
# 获取关键行（不能删除）
get_essential_lines() {
    # 从 analysis.json 获取关键构造
    KEY_CONSTRUCTS=$(jq -r '.test_case.key_constructs[]?' analysis.json 2>/dev/null | tr '\n' '|')
    KEYWORDS=$(jq -r '.keywords[]?' analysis.json 2>/dev/null | tr '\n' '|')
    
    # 组合模式
    PATTERN="${KEY_CONSTRUCTS}${KEYWORDS}"
    PATTERN="${PATTERN%|}"  # 移除尾部 |
    
    if [ -n "$PATTERN" ]; then
        grep -n -E "$PATTERN" working.sv | cut -d: -f1
    fi
}

ESSENTIAL_LINES=$(get_essential_lines)
echo "Essential lines: $ESSENTIAL_LINES"

# 可删除的行类型
REMOVABLE_PATTERNS=(
    "^\s*//"           # 单行注释
    "^\s*\$display"    # display 语句
    "^\s*\$monitor"    # monitor 语句
    "^\s*assert"       # 断言（除非是关键构造）
    "^\s*cover"        # 覆盖
    "^\s*wire\s+"      # 未使用的 wire 声明
    "^\s*reg\s+"       # 未使用的 reg 声明
    "^\s*logic\s+"     # 未使用的 logic 声明
)

# 迭代删除
TOTAL_LINES=$(wc -l < working.sv)
for ((i=TOTAL_LINES; i>=1; i--)); do
    # 跳过关键行
    if echo "$ESSENTIAL_LINES" | grep -qw "$i"; then
        continue
    fi
    
    # 获取当前行
    LINE=$(sed -n "${i}p" working.sv)
    
    # 跳过空行
    if [ -z "$(echo "$LINE" | tr -d '[:space:]')" ]; then
        continue
    fi
    
    # 跳过模块声明行
    if echo "$LINE" | grep -qE '^(module|endmodule|interface|endinterface)'; then
        continue
    fi
    
    echo "Trying to remove line $i: ${LINE:0:50}..."
    
    # 备份并删除
    cp working.sv working.sv.bak
    sed -i "${i}d" working.sv
    
    # 验证
    eval $REPRO_CMD > test_output.log 2>&1
    
    if grep -qE '(Assertion.*failed|LLVM ERROR)' test_output.log; then
        NEW_ASSERTION=$(grep -E '(Assertion.*failed|LLVM ERROR)' test_output.log | head -1)
        if [ "$NEW_ASSERTION" = "$ORIGINAL_ASSERTION" ]; then
            echo "  ✓ Line removed"
        else
            echo "  ✗ Different crash, restoring"
            cp working.sv.bak working.sv
        fi
    else
        echo "  ✗ No crash, restoring"
        cp working.sv.bak working.sv
    fi
done
```

### Phase 4: 清理

```bash
# 删除连续空行（保留单个空行）
sed -i '/^$/N;/^\n$/d' working.sv

# 删除行尾空白
sed -i 's/[[:space:]]*$//' working.sv

# 删除块注释 /* ... */
# 注意：简单实现，可能需要更复杂的处理
sed -i '/\/\*/,/\*\//d' working.sv

# 最终验证
eval $REPRO_CMD > final_test.log 2>&1
FINAL_ASSERTION=$(grep -E '(Assertion.*failed|LLVM ERROR)' final_test.log | head -1)

if [ "$FINAL_ASSERTION" != "$ORIGINAL_ASSERTION" ]; then
    echo "ERROR: Final assertion does not match original!"
    echo "Original: $ORIGINAL_ASSERTION"
    echo "Final: $FINAL_ASSERTION"
    exit 1
fi

echo "✓ Final verification passed"
```

### Phase 5: 生成输出

```bash
# 复制最终结果
cp working.$SOURCE_EXT bug.$SOURCE_EXT
cp final_test.log error.log

# 计算缩减比例
FINAL_LINES=$(wc -l < bug.$SOURCE_EXT)
REDUCTION=$(echo "scale=1; (1 - $FINAL_LINES / $ORIGINAL_LINES) * 100" | bc)

# 生成 command.txt（单行复现命令）
echo "$REPRO_CMD" | sed "s/$SOURCE_FILE/bug.$SOURCE_EXT/g" > command.txt

echo ""
echo "========================================"
echo "Minimization complete!"
echo "========================================"
echo "Original: $ORIGINAL_LINES lines"
echo "Minimized: $FINAL_LINES lines"
echo "Reduction: ${REDUCTION}%"
echo ""
echo "Output files:"
echo "  bug.$SOURCE_EXT - Minimized test case"
echo "  error.log - Error output"
echo "  command.txt - Reproduction command"
```

### 生成 minimize_report.md

```markdown
# Minimization Report

## Summary
- **Original file**: source.sv (156 lines)
- **Minimized file**: bug.sv (42 lines)
- **Reduction**: 73.1%
- **Crash preserved**: Yes

## Preservation Analysis

### Key Constructs Preserved
Based on `analysis.json`, the following constructs were kept:
- always_ff sensitivity list with array indexing
- packed union declaration

### Removed Elements
- Comments: 15 lines
- Unused wire declarations: 8 lines
- Unused module: `helper_module` (45 lines)
- Empty lines: 12 lines

## Verification

### Original Assertion
```
Assertion `isa<hw::SignalOp>(val)` failed
```

### Final Assertion
```
Assertion `isa<hw::SignalOp>(val)` failed
```

**Match**: ✅ Exact match

## Reproduction Command

```bash
circt-verilog bug.sv --ir-moore 2>&1 | firtool --verilog
```

## Notes
- Module `helper_module` was removed as it was not involved in the crash path
- The `$display` statements were removed as they are not essential for reproduction
- Wire `unused_wire` was removed as it had no connection to the crash
```

## SystemVerilog 特殊处理

### 可安全删除的元素

| 元素 | 条件 |
|------|------|
| `$display`, `$monitor` 语句 | 总是可删除 |
| `$dumpfile`, `$dumpvars` | 总是可删除 |
| 单行注释 `//` | 总是可删除 |
| 块注释 `/* */` | 总是可删除 |
| 属性 `(* ... *)` | 通常可删除，除非影响综合 |
| `initial` 块 | 如果只包含 display |
| 未使用的参数 | 需要验证不影响 |
| 未使用的端口 | 需要验证不影响 |

### 必须保留的元素

| 元素 | 原因 |
|------|------|
| 模块声明 | 语法必需 |
| 触发崩溃的构造 | 核心问题 |
| 类型定义（如果被引用） | 语法依赖 |
| 参数（如果被使用） | 功能依赖 |

## FIRRTL 特殊处理

### 可安全删除的元素

```
- printf 语句
- 注释
- 未使用的节点
- 未使用的 wire
- 未使用的寄存器（除非是崩溃原因）
```

### FIRRTL 最小化脚本

```bash
# FIRRTL 最小化通常更直接
# 因为 FIRRTL 语法更规则

# 删除 printf
sed -i '/printf/d' working.fir

# 删除 skip
sed -i '/skip/d' working.fir

# 删除注释
sed -i '/;/d' working.fir
```

## 验证要求

完成最小化前必须确认：

- [ ] `bug.sv` 存在且非空
- [ ] `error.log` 包含与原始相同的 assertion
- [ ] `command.txt` 包含可执行的单行命令
- [ ] 执行 `command.txt` 中的命令确实产生崩溃
- [ ] assertion message 完全匹配

## 完成后清理

```bash
# 验证成功后，删除原始文件
if [ -f bug.sv ] && [ -f error.log ]; then
    rm -f source.sv error.txt working.sv working.sv.bak test_output.log final_test.log
    echo "Cleaned up intermediate files"
fi
```

## 注意事项

1. **保守原则**：不确定时保留代码
2. **每步验证**：每次删除后都要验证崩溃
3. **签名匹配**：assertion message 必须完全匹配
4. **关键构造**：从 `analysis.json` 获取的关键构造绝不删除
5. **语法正确性**：最小化结果必须是合法的 SV/FIRRTL/MLIR
6. **退出码检查**：确保崩溃时退出码非零
