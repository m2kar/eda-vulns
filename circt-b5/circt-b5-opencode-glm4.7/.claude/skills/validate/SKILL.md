---
name: validate
description: 验证 CIRCT 崩溃测例的有效性。检查语法合规性、特性支持状态、跨工具验证，分类为 Bug 报告或其他类型。
argument-hint: (无参数，在当前目录操作)
allowed-tools: Shell(circt-verilog:*), Shell(firtool:*), Shell(verilator:*), Shell(iverilog:*), Shell(slang:*), Shell(cat:*), Shell(ls:*), Shell(which:*), Read, Write, Grep
---

# Skill: 测例验证

## 功能描述

验证最小化测例的有效性，确定是否应该作为 Bug 报告提交。包括：
1. 语法检查（IEEE 1800-2005/2017 合规性）
2. 特性支持检查（CIRCT 已知限制）
3. 跨工具验证（Verilator, Icarus Verilog, Slang）
4. 分类判定（Bug / Feature Request / Not a Bug）

## 输入

当前目录必须包含：
- `bug.sv` (或 `.fir`/`.mlir`) - 最小化测例
- `analysis.json` - 根因分析数据

## 输出

- `validation.json` - 验证数据
- `validation.md` - 验证报告

## 验证流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    VALIDATION WORKFLOW                           │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: Syntax Validation                                       │
│  ├── Check IEEE 1800 compliance                                  │
│  ├── Identify language features used                             │
│  └── Note any non-standard constructs                            │
├─────────────────────────────────────────────────────────────────┤
│  Step 2: Feature Support Check                                   │
│  ├── Check CIRCT known limitations                               │
│  ├── Check unsupported features list                             │
│  └── Match against existing issues                               │
├─────────────────────────────────────────────────────────────────┤
│  Step 3: Cross-Tool Validation                                   │
│  ├── Verilator (if available)                                    │
│  ├── Icarus Verilog (if available)                               │
│  ├── Slang (if available)                                        │
│  └── Compare results                                             │
├─────────────────────────────────────────────────────────────────┤
│  Step 4: Classification                                          │
│  ├── Determine result type                                       │
│  ├── Generate recommendation                                     │
│  └── Write validation report                                     │
└─────────────────────────────────────────────────────────────────┘
```

## 分类结果

| 结果 | 描述 | 后续动作 |
|------|------|----------|
| `report` | 确认是 Bug，应该报告 | 继续 check-duplicates |
| `feature_request` | 使用了不支持但有效的 SV 特性 | 标记为 feature request |
| `existing_issue` | 匹配 CIRCT 已知限制 | 链接到已有 Issue |
| `not_a_bug` | 设计限制或预期行为 | 结束流程 |
| `invalid_testcase` | 测例本身有语法错误 | 结束流程 |

## 执行步骤

### Step 1: 语法验证

```bash
# 检测测例文件
BUG_FILE=""
for ext in sv fir mlir; do
    if [ -f bug.$ext ]; then
        BUG_FILE="bug.$ext"
        break
    fi
done

if [ -z "$BUG_FILE" ]; then
    echo "Error: No bug file found"
    exit 1
fi

echo "Validating: $BUG_FILE"

# 对于 SystemVerilog，使用 slang 进行语法检查（如果可用）
if [ "$BUG_FILE" = "bug.sv" ]; then
    if command -v slang &> /dev/null; then
        echo "Running slang syntax check..."
        slang --lint-only bug.sv > slang_syntax.log 2>&1
        SLANG_EXIT=$?
        
        if [ $SLANG_EXIT -eq 0 ]; then
            SYNTAX_STATUS="valid"
            echo "  ✓ Syntax valid (slang)"
        else
            SYNTAX_STATUS="invalid"
            echo "  ✗ Syntax errors found"
            cat slang_syntax.log
        fi
    else
        echo "slang not available, syntax check skipped"
        SYNTAX_STATUS="unchecked"
    fi
fi
```

### Step 2: 特性支持检查

```bash
# CIRCT 已知不支持的特性
KNOWN_UNSUPPORTED=(
    "class"           # SV classes
    "interface"       # SV interfaces (部分支持)
    "covergroup"      # 功能覆盖
    "program"         # 程序块
    "bind"            # 绑定指令
    "fork"            # fork-join
    "randcase"        # 随机 case
    "randsequence"    # 随机序列
    "constraint"      # 约束块
    "dist"            # 分布约束
    "clocking"        # 时钟块
    "checker"         # 检查器
    "property"        # 属性（部分支持）
    "sequence"        # 序列（部分支持）
)

# 检查测例是否使用了不支持的特性
UNSUPPORTED_FOUND=""
for feature in "${KNOWN_UNSUPPORTED[@]}"; do
    if grep -qE "\b${feature}\b" bug.sv 2>/dev/null; then
        UNSUPPORTED_FOUND="${UNSUPPORTED_FOUND}${feature} "
    fi
done

if [ -n "$UNSUPPORTED_FOUND" ]; then
    echo "Warning: Test case uses potentially unsupported features: $UNSUPPORTED_FOUND"
    FEATURE_STATUS="unsupported_features"
else
    FEATURE_STATUS="supported"
fi
```

### CIRCT 已知限制数据库

```bash
# 从 analysis.json 获取信息用于匹配
DIALECT=$(jq -r '.dialect // "unknown"' analysis.json 2>/dev/null)
FAILING_PASS=$(jq -r '.failing_pass // "unknown"' analysis.json 2>/dev/null)
KEYWORDS=$(jq -r '.keywords[]?' analysis.json 2>/dev/null | tr '\n' ' ')

echo "Dialect: $DIALECT"
echo "Failing pass: $FAILING_PASS"
echo "Keywords: $KEYWORDS"

# 已知限制列表（示例）
# 实际应该查询 GitHub Issues 或维护一个本地数据库
KNOWN_LIMITATIONS=(
    "Moore:array_in_sensitivity:Array elements in sensitivity lists not fully supported"
    "Moore:packed_union:Packed union lowering incomplete"
    "FIRRTL:async_reset:Async reset inference has edge cases"
)

MATCHED_LIMITATION=""
for limitation in "${KNOWN_LIMITATIONS[@]}"; do
    LIMIT_DIALECT=$(echo "$limitation" | cut -d: -f1)
    LIMIT_KEY=$(echo "$limitation" | cut -d: -f2)
    LIMIT_DESC=$(echo "$limitation" | cut -d: -f3)
    
    if [ "$DIALECT" = "$LIMIT_DIALECT" ]; then
        if echo "$KEYWORDS" | grep -qi "$LIMIT_KEY"; then
            MATCHED_LIMITATION="$LIMIT_DESC"
            echo "Matched known limitation: $LIMIT_DESC"
            break
        fi
    fi
done
```

### Step 3: 跨工具验证

```bash
echo ""
echo "========================================"
echo "Cross-tool validation"
echo "========================================"

CROSS_TOOL_RESULTS=""

# Verilator
if command -v verilator &> /dev/null; then
    echo "Testing with Verilator..."
    verilator --lint-only bug.sv > verilator.log 2>&1
    VERILATOR_EXIT=$?
    
    if [ $VERILATOR_EXIT -eq 0 ]; then
        echo "  ✓ Verilator: OK"
        CROSS_TOOL_RESULTS="${CROSS_TOOL_RESULTS}verilator:pass,"
    else
        # 区分语法错误和 lint 警告
        if grep -qE '(Error|error:)' verilator.log; then
            echo "  ✗ Verilator: Error"
            CROSS_TOOL_RESULTS="${CROSS_TOOL_RESULTS}verilator:error,"
        else
            echo "  ⚠ Verilator: Warnings"
            CROSS_TOOL_RESULTS="${CROSS_TOOL_RESULTS}verilator:warning,"
        fi
    fi
else
    echo "  - Verilator: Not available"
    CROSS_TOOL_RESULTS="${CROSS_TOOL_RESULTS}verilator:unavailable,"
fi

# Icarus Verilog
if command -v iverilog &> /dev/null; then
    echo "Testing with Icarus Verilog..."
    iverilog -g2012 -o /dev/null bug.sv > iverilog.log 2>&1
    IVERILOG_EXIT=$?
    
    if [ $IVERILOG_EXIT -eq 0 ]; then
        echo "  ✓ Icarus: OK"
        CROSS_TOOL_RESULTS="${CROSS_TOOL_RESULTS}icarus:pass,"
    else
        echo "  ✗ Icarus: Error"
        CROSS_TOOL_RESULTS="${CROSS_TOOL_RESULTS}icarus:error,"
    fi
else
    echo "  - Icarus: Not available"
    CROSS_TOOL_RESULTS="${CROSS_TOOL_RESULTS}icarus:unavailable,"
fi

# Slang (已在语法检查中运行)
if command -v slang &> /dev/null; then
    if [ -f slang_syntax.log ]; then
        if [ "$SYNTAX_STATUS" = "valid" ]; then
            echo "  ✓ Slang: OK (from syntax check)"
            CROSS_TOOL_RESULTS="${CROSS_TOOL_RESULTS}slang:pass,"
        else
            echo "  ✗ Slang: Error (from syntax check)"
            CROSS_TOOL_RESULTS="${CROSS_TOOL_RESULTS}slang:error,"
        fi
    fi
else
    echo "  - Slang: Not available"
    CROSS_TOOL_RESULTS="${CROSS_TOOL_RESULTS}slang:unavailable,"
fi

echo ""
echo "Cross-tool results: $CROSS_TOOL_RESULTS"
```

### Step 4: 分类判定

```bash
# 分类逻辑
determine_classification() {
    # 规则 1: 语法无效 → invalid_testcase
    if [ "$SYNTAX_STATUS" = "invalid" ]; then
        echo "invalid_testcase"
        return
    fi
    
    # 规则 2: 使用已知不支持特性且其他工具通过 → feature_request
    if [ -n "$UNSUPPORTED_FOUND" ]; then
        if echo "$CROSS_TOOL_RESULTS" | grep -q "pass"; then
            echo "feature_request"
            return
        fi
    fi
    
    # 规则 3: 匹配已知限制 → existing_issue
    if [ -n "$MATCHED_LIMITATION" ]; then
        echo "existing_issue"
        return
    fi
    
    # 规则 4: 其他工具也报错 → not_a_bug (可能是无效测例)
    # 但要区分：其他工具的错误是同样的问题还是不同的问题
    PASS_COUNT=$(echo "$CROSS_TOOL_RESULTS" | grep -o "pass" | wc -l)
    ERROR_COUNT=$(echo "$CROSS_TOOL_RESULTS" | grep -o "error" | wc -l)
    
    if [ $ERROR_COUNT -gt 0 ] && [ $PASS_COUNT -eq 0 ]; then
        # 所有工具都报错，可能测例有问题
        echo "not_a_bug"
        return
    fi
    
    # 规则 5: 默认认为是 Bug
    echo "report"
}

CLASSIFICATION=$(determine_classification)
echo ""
echo "Classification: $CLASSIFICATION"
```

### Step 5: 生成输出

#### validation.json

```bash
TIMESTAMP=$(date -Iseconds)

cat > validation.json << EOF
{
  "version": "1.0",
  "timestamp": "$TIMESTAMP",
  "input_file": "$BUG_FILE",
  "syntax_check": {
    "status": "$SYNTAX_STATUS",
    "tool": "slang"
  },
  "feature_support": {
    "status": "$FEATURE_STATUS",
    "unsupported_found": "$(echo $UNSUPPORTED_FOUND | xargs)"
  },
  "known_limitations": {
    "matched": $([ -n "$MATCHED_LIMITATION" ] && echo "true" || echo "false"),
    "description": $(echo "$MATCHED_LIMITATION" | jq -Rs .)
  },
  "cross_tool_validation": {
    "verilator": "$(echo $CROSS_TOOL_RESULTS | grep -oE 'verilator:[^,]+' | cut -d: -f2)",
    "icarus": "$(echo $CROSS_TOOL_RESULTS | grep -oE 'icarus:[^,]+' | cut -d: -f2)",
    "slang": "$(echo $CROSS_TOOL_RESULTS | grep -oE 'slang:[^,]+' | cut -d: -f2)"
  },
  "classification": {
    "result": "$CLASSIFICATION",
    "confidence": "high",
    "reason": "$(case $CLASSIFICATION in
        invalid_testcase) echo "Syntax check failed" ;;
        feature_request) echo "Uses unsupported but valid SV features" ;;
        existing_issue) echo "Matches known CIRCT limitation" ;;
        not_a_bug) echo "Other tools also report errors" ;;
        report) echo "Valid test case, unique crash in CIRCT" ;;
    esac)"
  }
}
EOF

echo ""
echo "validation.json created"
```

#### validation.md

```bash
cat > validation.md << 'ENDOFMD'
# Validation Report

## Summary

| Check | Result |
|-------|--------|
ENDOFMD

echo "| Syntax Check | $SYNTAX_STATUS |" >> validation.md
echo "| Feature Support | $FEATURE_STATUS |" >> validation.md
echo "| Known Limitations | $([ -n "$MATCHED_LIMITATION" ] && echo "matched" || echo "none") |" >> validation.md
echo "| **Classification** | **$CLASSIFICATION** |" >> validation.md

cat >> validation.md << ENDOFMD

## Syntax Validation

**Tool**: slang
**Status**: $SYNTAX_STATUS

$([ -f slang_syntax.log ] && echo '```' && head -20 slang_syntax.log && echo '```')

## Feature Support Analysis

**Unsupported features detected**: $([ -n "$UNSUPPORTED_FOUND" ] && echo "$UNSUPPORTED_FOUND" || echo "None")

### CIRCT Known Limitations

$([ -n "$MATCHED_LIMITATION" ] && echo "**Matched**: $MATCHED_LIMITATION" || echo "No known limitation matched.")

## Cross-Tool Validation

| Tool | Status | Notes |
|------|--------|-------|
| Verilator | $(echo $CROSS_TOOL_RESULTS | grep -oE 'verilator:[^,]+' | cut -d: -f2) | $([ -f verilator.log ] && head -3 verilator.log) |
| Icarus | $(echo $CROSS_TOOL_RESULTS | grep -oE 'icarus:[^,]+' | cut -d: -f2) | $([ -f iverilog.log ] && head -3 iverilog.log) |
| Slang | $(echo $CROSS_TOOL_RESULTS | grep -oE 'slang:[^,]+' | cut -d: -f2) | Syntax check |

## Classification

**Result**: \`$CLASSIFICATION\`

**Reasoning**:
$(case $CLASSIFICATION in
    invalid_testcase) echo "The test case has syntax errors and is not valid SystemVerilog/FIRRTL." ;;
    feature_request) echo "The test case uses valid SystemVerilog features that are not yet supported by CIRCT. This should be filed as a feature request rather than a bug report." ;;
    existing_issue) echo "This issue matches a known CIRCT limitation. Consider checking the existing issue for status and updates." ;;
    not_a_bug) echo "Multiple tools report errors for this test case. The issue may be with the test case itself rather than CIRCT." ;;
    report) echo "The test case is valid and causes a unique crash in CIRCT. This should be reported as a bug." ;;
esac)

## Recommendation

$(case $CLASSIFICATION in
    invalid_testcase) echo "Fix the test case syntax errors before reporting." ;;
    feature_request) echo "File as a feature request with the unsupported features noted." ;;
    existing_issue) echo "Check the existing issue for updates. Consider adding this test case as additional information if it provides new insight." ;;
    not_a_bug) echo "Do not report. Investigate why other tools also reject this test case." ;;
    report) echo "Proceed to check for duplicates and generate the bug report." ;;
esac)
ENDOFMD

echo "validation.md created"
```

## IEEE 1800 特性参考

### 常见问题特性

| 特性 | IEEE 标准 | CIRCT 支持 |
|------|-----------|------------|
| always_ff | 1800-2005 | ✓ |
| always_comb | 1800-2005 | ✓ |
| always_latch | 1800-2005 | ✓ |
| logic 类型 | 1800-2005 | ✓ |
| 枚举 | 1800-2005 | ✓ |
| 结构体 | 1800-2005 | ✓ |
| union | 1800-2005 | 部分 |
| interface | 1800-2005 | 部分 |
| class | 1800-2005 | ✗ |
| covergroup | 1800-2005 | ✗ |
| 断言 (SVA) | 1800-2005 | 部分 |

## 注意事项

1. **跨工具差异**：不同工具对 SV 标准支持程度不同，需要综合判断
2. **版本差异**：IEEE 1800-2005 vs 2017 有差异
3. **Lint vs Error**：区分 lint 警告和真正的语法错误
4. **工具可用性**：如果验证工具不可用，记录为 "unchecked"
5. **保守判定**：有疑问时倾向于 "report"，让人工复核
