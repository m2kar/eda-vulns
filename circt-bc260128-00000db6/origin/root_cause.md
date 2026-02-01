# CIRCT Crash Root Cause Analysis

## 崩溃概要

| 项目 | 值 |
|------|-----|
| **Testcase ID** | 260128-00000db6 |
| **崩溃类型** | Assertion Failure |
| **MLIR Dialect** | comb (Combinational Logic) |
| **崩溃位置** | `mlir::RewriterBase::eraseOp` (PatternMatch.cpp:156) |
| **触发函数** | `extractConcatToConcatExtract` (CombFolds.cpp:548) |

## 错误信息

```
Assertion `op->use_empty() && "expected 'op' to have no uses"' failed.
```

## 调用栈关键路径

```
#11 mlir::RewriterBase::eraseOp (PatternMatch.cpp:156)
#12 circt::replaceOpAndCopyNamehint (Naming.cpp:82)
#13 extractConcatToConcatExtract (CombFolds.cpp:548)
#14 circt::comb::ExtractOp::canonicalize (CombFolds.cpp:615)
#28 Canonicalizer::runOnOperation (Canonicalizer.cpp:65)
```

## 测例分析

### 源代码 (source.sv)

```systemverilog
module array_example(input logic a, input logic b);
  logic [1:0] arr;
  
  assign arr[0] = a;
  
  always_comb arr[1] = b;
endmodule
```

### 关键语言特性

1. **数组部分赋值**: 对 `arr[0]` 和 `arr[1]` 分别赋值
2. **混合赋值风格**: `assign` 语句 + `always_comb` 块
3. **位域访问**: 对打包数组的单独位进行赋值

## 根因分析

### 问题定位

崩溃发生在 MLIR Canonicalization pass 执行 `comb::ExtractOp` 的 canonicalize pattern 时。

### 代码路径分析

在 `CombFolds.cpp` 中的 `extractConcatToConcatExtract` 函数：

```cpp
// CombFolds.cpp:546-551
if (reverseConcatArgs.size() == 1) {
  replaceOpAndCopyNamehint(rewriter, op, reverseConcatArgs[0]);  // Line 547 - 崩溃点
} else {
  replaceOpWithNewOpAndCopyNamehint<ConcatOp>(
      rewriter, op, SmallVector<Value>(llvm::reverse(reverseConcatArgs)));
}
```

### 根因假设

**主要假设: 替换操作时目标 Value 仍有其他用户**

当测例中存在混合的 `assign` 和 `always_comb` 对同一数组的不同位进行赋值时，CIRCT 会生成涉及 `comb.concat` 和 `comb.extract` 操作的 IR。

在 canonicalization 过程中：
1. `ExtractOp::canonicalize` 检测到 `extract(lo, cat(a, b, c, ...))` 模式
2. 调用 `extractConcatToConcatExtract` 尝试简化
3. 当简化结果只有一个元素时 (`reverseConcatArgs.size() == 1`)
4. 调用 `replaceOpAndCopyNamehint` 用 `reverseConcatArgs[0]` 替换原始 `ExtractOp`
5. **问题**: `replaceOpAndCopyNamehint` 内部可能尝试擦除某个操作，但该操作仍被其他 IR 节点使用

**可能的具体场景**:
- `reverseConcatArgs[0]` 是原始 `ConcatOp` 的某个输入
- 该输入被多个操作引用（例如被其他 extract 操作使用）
- 在 greedy pattern rewrite 过程中，操作之间的使用关系未正确更新
- 导致尝试擦除一个仍有用户的操作

### IR 结构推测

```mlir
// 推测的中间 IR 结构
%a = hw.input : i1
%b = hw.input : i1
%concat = comb.concat %a, %b : i1, i1  // 组合数组
%extract0 = comb.extract %concat from 0 : (i2) -> i1  // 提取 arr[0]
%extract1 = comb.extract %concat from 1 : (i2) -> i1  // 提取 arr[1]
// 当 canonicalize extract 时，可能触发 use-after-replace 问题
```

## 影响范围

- **受影响功能**: Comb dialect canonicalization
- **触发条件**: 数组部分赋值 + 混合 assign/always_comb
- **严重程度**: 编译器崩溃 (High)

## 相关源文件

| 文件 | 说明 |
|------|------|
| `lib/Dialect/Comb/CombFolds.cpp` | ExtractOp canonicalization patterns |
| `lib/Support/Naming.cpp` | replaceOpAndCopyNamehint 实现 |
| `llvm/mlir/lib/IR/PatternMatch.cpp` | MLIR RewriterBase::eraseOp |

## 建议修复方向

1. 在 `extractConcatToConcatExtract` 中替换操作前，验证目标 Value 的所有用户
2. 使用 `replaceAllUsesWith` 而非直接擦除操作
3. 检查 `replaceOpAndCopyNamehint` 的语义是否正确处理多用户场景
4. 添加针对数组部分赋值 + 混合 assign/always_comb 的测试用例
