# 验证报告

## 概述

| 项目 | 结果 |
|------|------|
| **分类** | **report** (应报告的 Bug) |
| **有效性** | ✅ 有效测例 |
| **最小化比例** | 84.6% (13 → 2 行) |

## 语法有效性检查

✅ **通过**

- 测例使用标准 SystemVerilog 语法
- `inout wire` 是合法的双向端口声明
- 无语法错误

## 跨工具验证

| 工具 | 结果 | 说明 |
|------|------|------|
| **Verilator** | ✅ 通过 | 无错误或警告 |
| **Slang** | ✅ 通过 | `Build succeeded: 0 errors, 0 warnings` |
| **circt-verilog** | ✅ 通过 | 成功生成 MLIR IR |
| **arcilator** | ❌ 崩溃 | 退出码 134 (SIGABRT) |

### 关键发现

1. **Verilator 和 Slang 都接受此测例** - 说明 SystemVerilog 代码语法正确
2. **circt-verilog 成功生成 IR** - 前端处理正常
3. **只有 arcilator 崩溃** - 问题特定于 Arc 方言的 LowerState pass

## Bug 分析

### 错误信息
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
Assertion `succeeded(ConcreteT::verifyInvariants(...))' failed.
```

### 崩溃位置
- **文件**: `lib/Dialect/Arc/Transforms/LowerState.cpp`
- **行号**: 219
- **函数**: `ModuleLowering::run()` → `StateType::get()`

### 根本原因
1. `inout` 端口被 MooreToCore 转换为 `!llhd.ref<i1>` 类型
2. `StateType::verify()` 要求类型具有已知的 bit width
3. `computeLLVMBitWidth()` 不支持 `llhd::RefType`（引用/指针类型不应有固定位宽）
4. 验证失败导致断言触发，程序崩溃

### 影响
- **严重程度**: 高
- **影响范围**: 任何包含 `inout` 端口的 SystemVerilog 模块都无法使用 arcilator 处理

## 结论

**分类: report (应报告的 Bug)**

**理由**:
1. 测例是有效的 SystemVerilog 代码
2. 多个工具（Verilator、Slang）都能正确处理
3. 崩溃是 CIRCT 内部的断言失败，不是用户错误
4. 应该发出清晰的错误信息，而不是程序崩溃

**建议修复**:
在 LowerState pass 中添加对 `llhd::RefType` 的早期检测，并发出友好的错误信息说明 arcilator 不支持 `inout` 端口，而非触发断言失败。
