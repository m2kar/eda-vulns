# CIRCT GitHub Issue 重复检查报告

**检查时间**: 2026-02-01 10:58:11  
**测试用例**: 260129-00001645

---

## 📋 摘要

| 项目 | 值 |
|------|-----|
| 搜索执行 | 8 个查询 |
| 原始结果 | 62 个 Issue |
| 唯一 Issue | 58 个 |
| 最高相似度 | 100/100 |
| 推荐 | **review_existing** |

---

## 🎯 推荐结论

### 推荐操作: **审查现有 Issue**

**高度匹配**: Issue #9574 - **[Arc] Assertion failure when lowering inout ports in sequential logic**

- **相似度分数**: 100/100
- **URL**: https://github.com/llvm/circt/issues/9574
- **状态**: 审核中

### 推荐理由

当前测试用例涉及的 Bug 特征:
- 工具: `arcilator` (Arc 方言)
- 传递: `LowerState`
- 错误类型: Assertion 失败
- 关键类型: `!llhd.ref<i1>` (inout 端口)
- 错误消息: "state type must have a known bit width"

与 Issue #9574 的匹配度:
- ✅ 相同的 Arc 方言相关问题
- ✅ 相同的 inout 端口处理问题
- ✅ 相同的 LowerState 传递相关
- ✅ 相同的类型验证失败

---

## 📊 搜索结果分析

### 最相关的 Issue


#### 1. Issue #9574
- **标题**: [Arc] Assertion failure when lowering inout ports in sequential logic
- **相似度**: 100/100
- **状态**: [Arc] Assertion failure when lowering inout ports in sequential logic
- **URL**: https://github.com/llvm/circt/issues/9574
- **摘要**: ## Description  CIRCT crashes with an assertion failure when compiling SystemVerilog code that uses `inout` ports within `always_ff` blocks. The crash occurs in the Arc dialect's `LowerStatePass` when...

#### 2. Issue #9052
- **标题**: [circt-verilog] Import difference of results in arcilator failure with remaining llhd constant_time
- **相似度**: 55/100
- **状态**: [circt-verilog] Import difference of results in arcilator failure with remaining llhd constant_time
- **URL**: https://github.com/llvm/circt/issues/9052
- **摘要**: Input test case:  ```verilog module bug (     input logic wr_clk,     input logic wr_data,     output logic [1:0] mem ); `ifdef CASE_1   always_ff @(posedge (wr_clk)) begin       mem[0] <= wr_data;   ...

#### 3. Issue #8332
- **标题**: [MooreToCore] Support for StringType from moore to llvm dialect
- **相似度**: 45/100
- **状态**: [MooreToCore] Support for StringType from moore to llvm dialect
- **URL**: https://github.com/llvm/circt/issues/8332
- **摘要**: Hi! Now I try to add types and operators in sim to get the lowered operators in moore, and then lower them to llvm dialect, so that the corresponding dynamic size container can be implemented in arcil...

#### 4. Issue #9395
- **标题**: [circt-verilog][arcilator] Arcilator assertion failure
- **相似度**: 40/100
- **状态**: [circt-verilog][arcilator] Arcilator assertion failure
- **URL**: https://github.com/llvm/circt/issues/9395
- **摘要**: Hi, all! Let's look at this example on _Verilog_:  ``` module comb_assert(     input wire clk,     input wire resetn );     always @* begin         if (resetn) begin             assert (0);         en...

#### 5. Issue #6948
- **标题**: [Arcilator] Integration tests failures without check-circt
- **相似度**: 40/100
- **状态**: [Arcilator] Integration tests failures without check-circt
- **URL**: https://github.com/llvm/circt/issues/6948
- **摘要**: ``` ******************** Failed Tests (6):   CIRCT :: arcilator/JIT/basic.mlir   CIRCT :: arcilator/JIT/counter.mlir   CIRCT :: arcilator/JIT/err-not-found.mlir   CIRCT :: arcilator/JIT/err-not-...


---

## 🔍 原始 Bug 分析

### 崩溃信息
- **工具**: arcilator
- **方言**: arc
- **传递**: LowerState
- **严重性**: high

### 错误消息
```
state type must have a known bit width; got '!llhd.ref<i1>'
```

### 断言位置
- **文件**: lib/Dialect/Arc/Transforms/LowerState.cpp
- **行号**: 219
- **函数**: ModuleLowering::run()

### 根本原因
**类别**: unsupported_type  
**描述**: LowerState pass does not handle inout ports (llhd.ref types)  
**触发构造**: inout port declaration  
**问题类型**: !llhd.ref<i1>

#### 不支持的类型处理
当前 `computeLLVMBitWidth()` 支持的类型:
- `IntegerType`
- `seq::ClockType`
- `hw::ArrayType`
- `hw::StructType`

**缺失**: `llhd::RefType` (用于 inout 端口)

### 源代码分析
```verilog
inout logic c
```
- **文件**: source.sv
- **语言**: SystemVerilog
- **行号**: 6

---

## 💡 建议修复方案

### 方案 1: Graceful Rejection
Add explicit check and emit user-friendly error for inout ports

### 方案 2: Type Support
Extend computeLLVMBitWidth to handle llhd::RefType

### 方案 3: Preprocessing
Convert inout ports to input/output pairs before lowering


---

## 📝 重复性结论

### 分析

当前测试用例与 Issue #9574 在以下方面完全一致:

1. **工具链**: 都涉及 `arcilator` 和 Arc 方言
2. **失败点**: 都在 `LowerState` 转换传递中
3. **类型问题**: 都是关于 `llhd.ref` 类型的处理
4. **错误模式**: 都是类型验证失败的断言错误

### 建议

**立即行动**:
1. 检查 Issue #9574 的当前状态
2. 如果该 Issue 已解决，验证修复是否包含此测试用例
3. 如果该 Issue 未解决，可以添加此测试用例作为补充信息

**不创建新 Issue 的原因**:
- 这是一个已知的、已被追踪的问题
- 不必要地创建重复的 Issue 会增加维护负担
- 应该在现有 Issue 中继续讨论和解决

---

## 📦 搜索统计

- **总搜索查询**: 8
- **找到的 Issue**: 62
- **唯一 Issue 数**: 58
- **生成时间**: 2026-02-01T10:57:51.893647

---

*此报告由自动化 Bug 分析工具生成*
