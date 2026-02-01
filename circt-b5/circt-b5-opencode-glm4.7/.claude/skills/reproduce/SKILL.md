---
name: reproduce
description: 验证 CIRCT Bug 可复现。解析原始错误日志，提取命令，用当前工具链执行，比对崩溃签名。
argument-hint: (无参数，在当前目录操作)
allowed-tools: Shell(circt-verilog:*), Shell(firtool:*), Shell(circt-opt:*), Shell(arcilator:*), Shell(cat:*), Shell(ls:*), Shell(mkdir:*), Shell(head:*), Shell(tail:*), Shell(grep:*), Shell(sed:*), Shell(awk:*), Read, Write
---

# Skill: 复现 CIRCT Bug

## 功能描述

验证崩溃用例在当前 CIRCT 工具链上可复现。解析原始错误日志，提取原始命令，用当前工具链执行，比对崩溃签名确认 Bug 仍然存在。

## 输入

当前目录必须包含：
- `error.txt` - 原始崩溃日志（包含命令和 stack trace）
- `source.sv` (或 `.fir`/`.mlir`) - 触发崩溃的测例

## 输出

- `reproduce.log` - 复现输出
- `metadata.json` - 工作流元数据

## 执行步骤

### 步骤 1: 检查输入文件

```bash
# 检查必需文件
if [ ! -f error.txt ]; then
    echo "Error: error.txt not found"
    exit 1
fi

# 检测测例文件
SOURCE_FILE=""
for ext in sv fir mlir; do
    if [ -f source.$ext ]; then
        SOURCE_FILE="source.$ext"
        break
    fi
done

if [ -z "$SOURCE_FILE" ]; then
    echo "Error: No source file found (source.sv/fir/mlir)"
    exit 1
fi

echo "Input files found:"
echo "  Error log: error.txt"
echo "  Source file: $SOURCE_FILE"
```

### 步骤 2: 解析原始命令

从 `error.txt` 中提取原始命令：

```bash
# 常见命令模式
# 1. circt-verilog source.sv --ir-moore | firtool --verilog
# 2. firtool source.fir --verilog
# 3. circt-opt source.mlir -pass-pipeline="..."

# 提取命令行（通常在文件开头或 "Command:" 后）
ORIGINAL_CMD=$(head -20 error.txt | grep -E '(circt-verilog|firtool|circt-opt|arcilator)' | head -1)

# 如果没找到，尝试其他模式
if [ -z "$ORIGINAL_CMD" ]; then
    # 查找 fuzzer 输出格式
    ORIGINAL_CMD=$(grep -E '^COMMAND:' error.txt | sed 's/^COMMAND://' | head -1)
fi

if [ -z "$ORIGINAL_CMD" ]; then
    echo "Warning: Could not extract command from error.txt"
    echo "Will try default commands based on file type"
fi

echo "Original command: $ORIGINAL_CMD"
```

### 步骤 3: 构建复现命令

根据文件类型和提取的命令构建复现命令：

```bash
# 设置 CIRCT 工具路径
CIRCT_BIN="${CIRCT_BIN:-}"
if [ -n "$CIRCT_BIN" ]; then
    export PATH="$CIRCT_BIN:$PATH"
fi

# 检查工具可用性
TOOL=""
if command -v circt-verilog &> /dev/null; then
    TOOL="circt-verilog"
elif command -v firtool &> /dev/null; then
    TOOL="firtool"
fi

if [ -z "$TOOL" ]; then
    echo "Error: No CIRCT tool found in PATH"
    echo "Set CIRCT_BIN environment variable or add to PATH"
    exit 1
fi

# 获取工具版本
TOOL_VERSION=$($TOOL --version 2>&1 | head -1)
echo "Tool version: $TOOL_VERSION"

# 构建复现命令
case $SOURCE_FILE in
    *.sv)
        # SystemVerilog - 使用 circt-verilog
        if [ -n "$ORIGINAL_CMD" ] && echo "$ORIGINAL_CMD" | grep -q "circt-verilog"; then
            # 替换原始路径为当前文件
            REPRO_CMD=$(echo "$ORIGINAL_CMD" | sed 's|[^ ]*/[^ ]*\.sv|source.sv|g')
        else
            # 默认命令
            REPRO_CMD="circt-verilog source.sv --ir-moore 2>&1 | firtool --verilog 2>&1"
        fi
        ;;
    *.fir)
        # FIRRTL - 使用 firtool
        if [ -n "$ORIGINAL_CMD" ] && echo "$ORIGINAL_CMD" | grep -q "firtool"; then
            REPRO_CMD=$(echo "$ORIGINAL_CMD" | sed 's|[^ ]*/[^ ]*\.fir|source.fir|g')
        else
            REPRO_CMD="firtool source.fir --verilog 2>&1"
        fi
        ;;
    *.mlir)
        # MLIR - 使用 circt-opt 或 firtool
        if [ -n "$ORIGINAL_CMD" ]; then
            REPRO_CMD=$(echo "$ORIGINAL_CMD" | sed 's|[^ ]*/[^ ]*\.mlir|source.mlir|g')
        else
            REPRO_CMD="circt-opt source.mlir 2>&1"
        fi
        ;;
esac

echo "Reproduction command: $REPRO_CMD"
```

### 步骤 4: 执行复现

```bash
echo "========================================"
echo "Executing reproduction command..."
echo "========================================"

# 执行命令并捕获输出
eval $REPRO_CMD > reproduce.log 2>&1
EXIT_CODE=$?

echo "Exit code: $EXIT_CODE"
echo ""

# 显示输出摘要
echo "Output summary (first 50 lines):"
head -50 reproduce.log
```

### 步骤 5: 比对崩溃签名

```bash
# 提取原始 assertion message
ORIGINAL_ASSERTION=$(grep -E '(Assertion.*failed|LLVM ERROR|error:|fatal:)' error.txt | head -1)

# 提取复现 assertion message
REPRO_ASSERTION=$(grep -E '(Assertion.*failed|LLVM ERROR|error:|fatal:)' reproduce.log | head -1)

echo ""
echo "========================================"
echo "Crash signature comparison:"
echo "========================================"
echo "Original: $ORIGINAL_ASSERTION"
echo "Reproduced: $REPRO_ASSERTION"

# 比对
if [ $EXIT_CODE -eq 0 ]; then
    REPRODUCED="false"
    MATCH_RESULT="no_crash"
    echo "Result: No crash occurred (exit code 0)"
elif [ -z "$REPRO_ASSERTION" ]; then
    REPRODUCED="false"
    MATCH_RESULT="no_assertion"
    echo "Result: Crash but no recognizable assertion/error message"
elif [ "$ORIGINAL_ASSERTION" = "$REPRO_ASSERTION" ]; then
    REPRODUCED="true"
    MATCH_RESULT="exact_match"
    echo "Result: ✅ EXACT MATCH - Bug reproduced!"
elif echo "$REPRO_ASSERTION" | grep -qF "$(echo "$ORIGINAL_ASSERTION" | cut -c1-50)"; then
    REPRODUCED="true"
    MATCH_RESULT="partial_match"
    echo "Result: ✅ PARTIAL MATCH - Bug reproduced!"
else
    REPRODUCED="false"
    MATCH_RESULT="different_crash"
    echo "Result: ❌ Different crash - may be a different bug or fixed"
fi
```

### 步骤 6: 生成 metadata.json

```bash
TIMESTAMP=$(date -Iseconds)

cat > metadata.json << EOF
{
  "version": "1.0",
  "timestamp": "$TIMESTAMP",
  "input": {
    "error_file": "error.txt",
    "source_file": "$SOURCE_FILE"
  },
  "tool": {
    "name": "$TOOL",
    "version": "$TOOL_VERSION",
    "path": "$(which $TOOL)"
  },
  "reproduction": {
    "command": "$REPRO_CMD",
    "exit_code": $EXIT_CODE,
    "reproduced": $REPRODUCED,
    "match_result": "$MATCH_RESULT"
  },
  "crash_signature": {
    "original": $(echo "$ORIGINAL_ASSERTION" | jq -Rs .),
    "reproduced": $(echo "$REPRO_ASSERTION" | jq -Rs .)
  },
  "files": {
    "reproduce_log": "reproduce.log"
  }
}
EOF

echo ""
echo "metadata.json created:"
cat metadata.json | jq .
```

## 返回信息

### 复现成功

```
✅ Bug 已复现

工具版本: firtool-1.140.0
复现命令: circt-verilog source.sv --ir-moore | firtool --verilog
崩溃签名: Assertion `isa<X>(Val)` failed

匹配结果: exact_match
日志: reproduce.log
元数据: metadata.json

下一步: 运行 /root-cause-analysis skill 进行根因分析
```

### 复现失败

```
❌ 未能复现 Bug

工具版本: firtool-1.140.0
复现命令: circt-verilog source.sv --ir-moore | firtool --verilog
结果: 程序正常完成（exit code 0）

可能原因:
1. Bug 已在当前版本修复
2. 需要特定的工具版本或参数
3. 测例文件不完整

建议: 检查原始错误日志中的工具版本和参数
```

## 注意事项

1. **工具路径**：如果 CIRCT 工具不在 PATH 中，设置 `CIRCT_BIN` 环境变量
2. **版本差异**：不同版本的命令行参数可能不同
3. **管道命令**：`circt-verilog | firtool` 需要正确处理管道错误
4. **崩溃签名**：assertion message 是最可靠的签名，stack trace 可能因编译选项不同而变化
5. **部分匹配**：如果 assertion message 前 50 字符匹配，也认为是复现成功
