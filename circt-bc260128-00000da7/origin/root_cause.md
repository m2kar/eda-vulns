# CIRCT Arcilator Timeout Root Cause Analysis

## 问题描述

**Testcase ID**: 260128-00000da7  
**Crash Type**: Timeout (300s)  
**Affected Tool**: arcilator

### 复现命令

```bash
circt-verilog --ir-hw source.sv | arcilator | opt -O0 | llc -O0 --filetype=obj -o output.o
```

## 代码分析

### 问题构造

```systemverilog
module array_example(
  input logic [2:0] idx,
  input logic clk,
  output logic result
);

  typedef struct packed {
    logic valid;
    logic [7:0] data;
  } array_elem_t;

  array_elem_t arr [0:7];   // 8-element unpacked array of packed structs

  always_comb begin
    arr[idx].data = 8'hFF;   // Line 15: WRITE to arr[idx]
    
    if (arr[idx].valid) begin // Line 17: READ from arr[idx]
      result = 1'b1;
    end else begin
      result = 1'b0;
    end
  end

endmodule
```

### 关键问题点

1. **Unpacked Array with Packed Struct Elements**:
   - `array_elem_t arr [0:7]` 定义了一个包含 8 个 packed struct 元素的 unpacked 数组
   - 每个元素包含 `valid` (1-bit) 和 `data` (8-bit) 字段

2. **Same-Index Write-then-Read in always_comb**:
   - 第 15 行: `arr[idx].data = 8'hFF` - 对 `arr[idx]` 的 `data` 字段写入
   - 第 17 行: `if (arr[idx].valid)` - 读取 `arr[idx]` 的 `valid` 字段
   - **问题**: 同一个组合逻辑块中，对同一个动态索引的数组元素先写后读

3. **动态索引导致的依赖分析困难**:
   - `idx` 是运行时输入，编译器无法静态确定访问的是哪个数组元素
   - 这导致依赖分析器可能将所有元素都视为潜在的读写冲突

## 根因假设

### 假设 1: ConvertToArcs 中的组合循环检测无限循环

在 `lib/Conversion/ConvertToArcs/ConvertToArcs.cpp` 中，存在组合循环检测逻辑：

```cpp
// Line 180-183
if (!seen.insert(definingOp).second) {
  definingOp->emitError("combinational loop detected");
  return failure();
}
```

当处理动态索引的数组访问时：
- 对 `arr[idx].data` 的写入和 `arr[idx].valid` 的读取可能被视为同一操作的输入/输出
- 依赖图可能形成一个复杂的循环结构
- 如果检测逻辑未能正确识别这种边界情况，可能导致无限遍历

### 假设 2: LowerState Pass 中的工作列表无限循环

在 `lib/Dialect/Arc/Transforms/LowerState.cpp` 中：

```cpp
// Line 286-290
if (!opsSeen.insert({defOp, phase}).second) {
  defOp->emitOpError("is on a combinational loop");
  dumpWorklist();
  return failure();
}
```

该代码检测组合循环，但如果数组元素的依赖关系被错误地建模：
- 工作列表可能不断添加相同的操作
- `opsSeen` 检查可能因为 phase 不同而未能触发
- 导致工作列表无限增长或无限循环

### 假设 3: 数组元素粒度的依赖追踪缺陷

Arcilator 可能缺乏对 unpacked 数组中不同字段的精细依赖追踪：
- `arr[idx].data` 写入被视为对整个 `arr[idx]` 的修改
- `arr[idx].valid` 读取被视为依赖整个 `arr[idx]`
- 形成假性循环依赖: `write(arr[idx]) -> read(arr[idx]) -> output -> (某个反馈路径)`

## 相关组件

| 组件 | 文件 | 可疑度 |
|------|------|--------|
| ConvertToArcs | `lib/Conversion/ConvertToArcs/ConvertToArcs.cpp` | 高 |
| LowerState | `lib/Dialect/Arc/Transforms/LowerState.cpp` | 高 |
| SplitLoops | `lib/Dialect/Arc/Transforms/SplitLoops.cpp` | 中 |
| ArcCanonicalizerPass | `lib/Dialect/Arc/Transforms/ArcCanonicalizer.cpp` | 低 |

## 建议修复方向

1. **增强数组字段级别的依赖分析**:
   - 区分对同一数组元素不同字段的读写
   - `arr[idx].data` 写入不应该阻塞 `arr[idx].valid` 读取

2. **添加无限循环检测和超时**:
   - 在依赖遍历中添加最大迭代次数限制
   - 超过限制时生成有意义的错误消息

3. **改进动态索引处理**:
   - 对于动态索引的数组访问，采用保守但有界的依赖分析策略

## 参考

- CIRCT Issue 相关: 搜索 "combinational loop" 或 "arcilator timeout"
- 相关错误处理代码位于 `LowerState.cpp:287` 和 `ConvertToArcs.cpp:181`
