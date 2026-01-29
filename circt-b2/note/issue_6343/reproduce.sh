#!/bin/bash
# Bug 复现脚本 - Issue #6343
# 描述: circt-opt 的 --lower-scf-to-calyx pass 在处理包含 func.call 和嵌套循环的 MLIR 代码时会触发断言失败
# 用法: ./reproduce.sh <firtool_version>
# 示例: ./reproduce.sh 1.54.0

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VERSION="${1:-1.54.0}"  # 默认版本
TOOL_DIR="$PROJECT_ROOT/builds/firtool-$VERSION/bin"

# 检查版本是否存在
if [ ! -d "$TOOL_DIR" ]; then
    echo "Error: Version $VERSION not found in $PROJECT_ROOT/builds/"
    echo "Available versions:"
    ls "$PROJECT_ROOT/builds/" | grep firtool | sed 's/firtool-/  /'
    exit 1
fi

echo "========================================"
echo "Testing Issue #6343 with version: $VERSION"
echo "========================================"
echo ""

# 创建结果目录
RESULT_DIR="$SCRIPT_DIR/result"
mkdir -p "$RESULT_DIR/output"

# 复现步骤（根据 Issue 中的命令链）
echo "Step 1: Running mlir-opt --lower-affine..."
"$TOOL_DIR/mlir-opt" --lower-affine "$SCRIPT_DIR/test.mlir" > "$RESULT_DIR/output/lowered1.mlir" 2>&1 || echo "Step 1 failed or completed with errors"

echo ""
echo "Step 2: Running mlir-opt --scf-for-to-while..."
"$TOOL_DIR/mlir-opt" --scf-for-to-while "$RESULT_DIR/output/lowered1.mlir" > "$RESULT_DIR/output/lowered2.mlir" 2>&1 || echo "Step 2 failed or completed with errors"

echo ""
echo "Step 3: Running circt-opt --lower-scf-to-calyx..."
# 使用 || true 防止错误导致脚本退出，我们需要捕获错误
# 使用 pass-pipeline 格式，将 top-level-function 放在 pass 的花括号内
"$TOOL_DIR/circt-opt" --pass-pipeline='builtin.module(lower-scf-to-calyx{top-level-function=mlir_func_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E7conv_2D},canonicalize)' "$RESULT_DIR/output/lowered2.mlir" 2>&1 | tee "$RESULT_DIR/run_v${VERSION}.log" || true

echo ""
echo "========================================"
echo "Test completed. Check results in:"
echo "  $RESULT_DIR/run_v${VERSION}.log"
echo "  $RESULT_DIR/output/"
echo "========================================"
