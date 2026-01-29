# Root Cause Analysis Report

## Executive Summary

Arcilator 的 `LowerState` pass 在处理包含 `inout` 双向端口的设计时崩溃。崩溃原因是 LLHD 方言的引用类型 `!llhd.ref<i1>` 无法转换为 `arc::StateType`，因为 `StateType` 要求类型具有已知位宽，而 `llhd.ref` 是引用类型。

## Crash Context

| 属性 | 值 |
|------|-----|
| **Tool** | arcilator |
| **Dialect** | Arc (处理 LLHD 输入) |
| **Failing Pass** | LowerStatePass |
| **Crash Type** | Assertion failure |
| **CIRCT Version** | 1.139.0 |

## Error Analysis

### Assertion Message

```
state type must have a known bit width; got '!llhd.ref<i1>'
```

### Assertion Location

```
StorageUniquerSupport.h:180: Assertion `succeeded(ConcreteT::verifyInvariants(...))' failed
```

### Key Stack Frames

| # | Function | Location |
|---|----------|----------|
| #12 | `circt::arc::StateType::get(mlir::Type)` | `ArcTypes.cpp.inc:108` |
| #13 | `(anonymous namespace)::ModuleLowering::run()` | `LowerState.cpp:219` |
| #15 | `(anonymous namespace)::LowerStatePass::runOnOperation()` | `LowerState.cpp:1198` |

## Test Case Analysis

### Code Summary

测例定义了一个带有混合端口类型的模块 `MixedPorts`：
- 标准输入端口 (`clk`, `a`, `dir`)
- 标准输出端口 (`b`)
- **双向 inout 端口 (`c`)** - 触发崩溃的关键构造

### Source Code

```systemverilog
module MixedPorts(
  input logic clk,
  input logic a,
  input logic dir,
  output logic b,
  inout wire c
);

  logic data_in;
  
  // Data input assignment
  assign data_in = a;
  
  // Registered output with clocked always block
  always_ff @(posedge clk) begin
    b <= data_in;
  end
  
  // Tri-state driver for inout port
  assign c = (dir) ? data_in : 1'bz;

endmodule
```

### Key Constructs

1. **`inout wire c`** - 双向端口声明
2. **`always_ff @(posedge clk)`** - 时序逻辑块
3. **`assign c = (dir) ? data_in : 1'bz`** - 三态驱动器

### Problematic Patterns

- **双向端口 (inout)**: 在 LLHD 方言中表示为 `llhd.ref<T>` 引用类型
- **三态赋值 (`1'bz`)**: 需要特殊的信号语义支持

## CIRCT Source Analysis

### Crash Location

- **File**: `lib/Dialect/Arc/Transforms/LowerState.cpp`
- **Line**: 219
- **Function**: `ModuleLowering::run()`

### Processing Path

```
1. circt-verilog --ir-hw source.sv
   └─> 生成包含 llhd.ref 类型的 IR (用于 inout 端口)

2. arcilator 接收 IR
   └─> LowerStatePass 开始执行
       └─> ModuleLowering::run() 处理模块
           └─> 尝试为 llhd.ref<i1> 创建 StateType
               └─> StateType::get() 调用 verifyInvariants()
                   └─> [FAILS] "state type must have a known bit width"
```

### Type Incompatibility

| 类型 | 描述 | 位宽信息 |
|------|------|----------|
| `!llhd.ref<i1>` | LLHD 引用类型 | 无直接位宽 (是引用) |
| `!arc.state<i1>` | Arc 状态类型 | 要求已知位宽 |

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)

**Cause**: Arcilator 的 `LowerState` pass 不支持 LLHD 引用类型

**Description**: 
当 `circt-verilog --ir-hw` 处理含有 `inout` 端口的 SystemVerilog 设计时，会生成包含 `!llhd.ref<T>` 类型的 IR。Arcilator 的 `LowerState` pass 试图将所有端口/信号转换为 `arc::StateType`，但没有对 `llhd.ref` 类型进行特殊处理。`StateType::verifyInvariants()` 检查类型必须有已知位宽，而 `llhd.ref` 作为引用类型不满足此条件。

**Evidence**:
1. 错误消息明确指出 "state type must have a known bit width; got '!llhd.ref<i1>'"
2. 测例包含 `inout wire c` 双向端口
3. Stack trace 显示崩溃发生在 `StateType::get()` 调用 `verifyInvariants()` 时
4. LLHD 的 `ref` 类型用于表示可被多个驱动器驱动的信号引用

**Mechanism**:
```
inout wire c  →  !llhd.ref<i1>  →  StateType::get(!llhd.ref<i1>)  →  CRASH
```

### Hypothesis 2 (Medium Confidence)

**Cause**: `circt-verilog --ir-hw` 输出格式不完全兼容 arcilator

**Description**:
`--ir-hw` 标志应该生成纯 HW 方言输出，但 `inout` 端口可能需要 LLHD 的引用语义才能正确表示。这导致输出的 IR 包含 arcilator 无法处理的混合方言类型。

**Evidence**:
1. 命令链使用 `circt-verilog --ir-hw | arcilator`
2. HW 方言可能没有原生的 `inout` 表示
3. 必须借用 LLHD 的 `ref` 类型来表示双向端口

### Hypothesis 3 (Low Confidence)

**Cause**: 三态值 (`1'bz`) 处理缺失

**Description**:
除了 `inout` 端口本身，三态赋值 `assign c = (dir) ? data_in : 1'bz` 也可能导致问题，因为 arcilator 可能不支持高阻态值的仿真。

**Evidence**:
1. 测例使用 `1'bz` 高阻态值
2. Arcilator 设计用于快速仿真，可能不支持所有 Verilog 语义

## Suggested Fix Directions

1. **在 LowerState pass 中添加 `llhd.ref` 类型检测**
   - 如果遇到 `llhd.ref` 类型，生成友好的错误消息而非断言失败
   - 或者提取引用的基础类型 (`llhd.ref<i1>` → `i1`) 并处理

2. **在 arcilator 中添加对 inout 端口的支持**
   - 实现 LLHD 引用类型到 Arc 状态的转换逻辑
   - 可能需要特殊的仿真模型来处理双向端口

3. **改进 circt-verilog 的 --ir-hw 输出**
   - 确保输出不包含 LLHD 类型，或者
   - 文档说明某些构造需要不同的输出模式

4. **添加前置验证 pass**
   - 在 arcilator 流水线开始时检查不支持的类型
   - 提前报错，避免深层 pass 中的断言失败

## Keywords for Issue Search

`arcilator` `inout` `llhd.ref` `StateType` `LowerState` `bit width` `bidirectional port` `tri-state`

## Related Files

| Path | Reason |
|------|--------|
| `lib/Dialect/Arc/Transforms/LowerState.cpp` | 崩溃发生位置 |
| `include/circt/Dialect/Arc/ArcTypes.td` | StateType 定义 |
| `include/circt/Dialect/LLHD/IR/LLHDTypes.td` | llhd.ref 类型定义 |
| `tools/arcilator/arcilator.cpp` | arcilator 主程序 |
| `lib/Tools/circt-verilog/` | circt-verilog 实现 |
