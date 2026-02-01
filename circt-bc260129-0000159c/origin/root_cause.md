# Root Cause Analysis Report

## 崩溃概述

**错误类型**: Assertion Failure  
**崩溃位置**: `circt::arc::StateType::get()` → `verifyInvariants()` 失败  
**错误信息**: `state type must have a known bit width; got '!llhd.ref<i1>'`

## 触发条件

该崩溃由以下 SystemVerilog 构造组合触发：

1. **inout 端口声明**: `inout logic c` - 双向端口
2. **时序逻辑**: `always_ff @(posedge clk)` 块
3. **条件三态赋值**: `assign c = (a) ? temp_reg[0] : 1'bz`

## 技术分析

### 1. 错误产生机制

问题发生在 `arcilator` 工具的 `LowerState` pass 中。该 pass 负责将 HW/LLHD 方言转换为 Arc 方言的状态表示。

**关键代码路径**:
```
LowerStatePass::runOnOperation()
  → ModuleLowering::run()
    → RootInputOp::create() 
      → StateType::get(arg.getType())
        → StateType::verify()  // FAILS
```

### 2. 类型系统不兼容

**Arc StateType 的约束** (来自 `ArcTypes.cpp`):
```cpp
LogicalResult StateType::verify(..., Type innerType) {
  if (!computeLLVMBitWidth(innerType))
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

`computeLLVMBitWidth()` 函数只支持以下类型：
- `seq::ClockType` → 1 bit
- `IntegerType` → 直接宽度
- `hw::ArrayType` → 递归计算
- `hw::StructType` → 递归计算

**LLHD RefType 不在支持列表中**，因此 `computeLLVMBitWidth()` 返回 `std::nullopt`，导致验证失败。

### 3. inout 端口的类型转换路径

当 SystemVerilog 的 `inout` 端口被转换时：

1. **circt-verilog (ImportVerilog)**: 解析 `inout logic c` 
2. **MooreToCore 转换**: 将 Moore 方言转为 LLHD/HW 方言
   - `inout` 端口被表示为 `!llhd.ref<i1>` 类型
   - 这是一个引用类型，用于表示可被驱动的信号
3. **arcilator LowerState**: 尝试为所有模块参数创建状态存储
   ```cpp
   // LowerState.cpp:218-220
   for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
     auto state = RootInputOp::create(..., StateType::get(arg.getType()), ...);
   }
   ```

### 4. 根本原因

**Arc 方言的 StateType 设计上不支持 LLHD RefType**。

这是两个方言之间的语义不匹配：
- **LLHD RefType**: 表示一个可以被读取和驱动的信号引用，是 LLHD 方言用于建模硬件信号的核心类型
- **Arc StateType**: 表示模拟器中的状态存储，需要已知的比特宽度来分配内存

`inout` 端口产生的 `!llhd.ref<i1>` 类型无法直接映射为 Arc 状态，因为：
1. RefType 是一个引用/指针类型，不是值类型
2. `computeLLVMBitWidth()` 没有处理 RefType 的逻辑

### 5. MixedPorts 模块的触发作用

测试用例中的模块结构：
```systemverilog
module MixedPorts(
  input  logic clk,
  input  logic a,
  output logic b,
  inout  logic c   // <-- 问题根源
);
```

- `clk`, `a` 是 input 端口 → 正常转换为值类型
- `b` 是 output 端口 → 正常处理
- `c` 是 inout 端口 → 被转换为 `!llhd.ref<i1>`，在 LowerState 中触发断言

## 崩溃堆栈解析

```
#11 circt::arc::StateType::get(mlir::Type)
    → 尝试创建 StateType<llhd.ref<i1>>，触发 verifyInvariants 断言

#12 ModuleLowering::run() [LowerState.cpp:219]
    → 为模块输入参数分配状态存储时调用 StateType::get()

#13 LowerStatePass::runOnOperation() [LowerState.cpp:1198]
    → Pass 入口点
```

## 相关代码位置

| 文件 | 行号 | 描述 |
|------|------|------|
| `lib/Dialect/Arc/ArcTypes.cpp` | 29-76 | `computeLLVMBitWidth()` 函数实现 |
| `lib/Dialect/Arc/ArcTypes.cpp` | 80-87 | `StateType::verify()` 验证函数 |
| `lib/Dialect/Arc/Transforms/LowerState.cpp` | 215-221 | 为输入端口创建状态的代码 |
| `lib/Dialect/Arc/ArcOps.cpp` | 338-339 | ModelOp 验证中检查 inout 不支持 |
| `include/circt/Dialect/LLHD/LLHDTypes.td` | 34-48 | RefType 定义 |

## 问题分类

- **Bug 类型**: 跨方言类型系统不兼容
- **严重程度**: 中等 - 工具崩溃，但有明确的不支持构造
- **影响范围**: 使用 `inout` 端口的 SystemVerilog 代码无法通过 arcilator 处理

## 可能的修复方向

1. **早期检测与友好报错**: 在 LowerState pass 开始前检测 LLHD RefType 参数并发出可理解的错误信息，而非断言失败
2. **扩展 StateType 支持**: 在 `computeLLVMBitWidth()` 中添加对 RefType 的处理（提取内部类型的宽度）
3. **方言转换**: 在进入 Arc 方言之前，将 RefType 转换为其他表示形式
4. **文档化限制**: 明确记录 arcilator 不支持 inout 端口

## 最小化测试用例

```systemverilog
module MixedPorts(
  input  logic clk,
  input  logic a,
  output logic b,
  inout  logic c
);
  logic [3:0] temp_reg;
  always_ff @(posedge clk) begin
    for (int i = 0; i < 4; i++) begin
      temp_reg[i] = a & i[0];
    end
  end
  assign b = temp_reg[0];
  assign c = (a) ? temp_reg[0] : 1'bz;
endmodule
```

复现命令:
```bash
circt-verilog --ir-hw source.sv | arcilator
```
