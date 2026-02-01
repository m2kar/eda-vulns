# CIRCT 超时崩溃根因分析报告

## 1. 崩溃概述

- **崩溃类型**: Timeout (编译超时)
- **超时时间**: 300 秒
- **测试用例 ID**: 260129-000018aa
- **方言**: Moore (SystemVerilog) → Arc/HW

## 2. 编译管道

```
circt-verilog --ir-hw source.sv | arcilator | opt -O0 | llc -O0
```

编译流程:
1. `circt-verilog --ir-hw`: 将 SystemVerilog 解析为 HW/Moore IR
2. `arcilator`: 将 HW IR 转换为 Arc 方言，用于仿真
3. `opt -O0` / `llc -O0`: LLVM 后端优化和代码生成

## 3. 测例代码分析

```systemverilog
module top;
  NestedA inst_a();
endmodule

module NestedA;
  module NestedB;           // 嵌套模块声明 (层级 1)
    module NestedC;         // 嵌套模块声明 (层级 2)
      logic [7:0] data;
      
      function automatic bit func2(input bit y);
        func2 = ~y;
      endfunction
      
      function automatic bit func1(input bit x);
        func1 = func2(x);   // 函数调用链
      endfunction
      
      always_comb data[0] = func1(data[7]);  // 组合逻辑循环
    endmodule
    
    NestedC inst_c();
  endmodule
  
  NestedB inst_b();
endmodule
```

### 3.1 关键问题构造

#### 问题 1: 三级嵌套模块
- `NestedA` → `NestedB` → `NestedC`
- SystemVerilog 允许嵌套模块声明（nested module），但这是较少使用的特性
- 编译器需要处理嵌套作用域和符号解析

#### 问题 2: 函数调用链
- `func1()` 调用 `func2()`
- 两个函数都使用 `automatic` 存储类
- 编译器在 `always_comb` 上下文中需要内联这些函数

#### 问题 3: 组合逻辑循环依赖
```
always_comb data[0] = func1(data[7]);
```
- `data[0]` 的值依赖于 `data[7]`
- 在 `always_comb` 块中，信号变化会触发重新计算
- 虽然这不是真正的组合环（`data[0]` 和 `data[7]` 是不同位），但编译器可能在分析时保守处理

## 4. 潜在超时原因分析

### 4.1 假设 1: 嵌套模块展开复杂度

在 `ImportVerilog` 阶段，编译器需要:
1. 解析嵌套模块声明
2. 为每个嵌套模块创建独立的模块定义
3. 处理嵌套作用域的符号查找

CIRCT 源码中的 `convertModuleHeader()` 和 `convertModuleBody()` 函数处理模块转换，但嵌套模块可能导致:
- 重复的模块定义比较
- 参数匹配检查的指数复杂度

### 4.2 假设 2: Arc 方言的循环分析

`arcilator` 使用 Arc 方言进行仿真转换。在 `ConvertToArcs.cpp` 中:
- `analyzeFanIn()` 函数计算每个操作的扇入掩码
- 使用位掩码表示操作依赖关系
- 当存在复杂的嵌套结构时，工作列表算法可能陷入长时间迭代

关键代码段:
```cpp
while (!worklist.empty()) {
    // ... 迭代处理操作数
    if (!seen.insert(definingOp).second) {
        definingOp->emitError("combinational loop detected");
        return failure();
    }
}
```

### 4.3 假设 3: SplitLoops Pass 的复杂度

`SplitLoops.cpp` 中的 `ensureNoLoops()` 函数使用深度优先搜索检测零延迟循环:
- 工作列表算法遍历所有操作
- 对于嵌套模块展开后的大量操作，可能导致长时间运行

### 4.4 假设 4: MooreToCore 转换中的 always_comb 处理

`MooreToCore.cpp` 中的 `ProcedureOpConversion` 处理 `always_comb`:
```cpp
if (op.getKind() == ProcedureKind::AlwaysComb ||
    op.getKind() == ProcedureKind::AlwaysLatch) {
    // 收集所有需要观察的值
    getValuesToObserve(&op.getBody(), setInsertionPoint, typeConverter,
                       rewriter, observedValues);
}
```

这个过程需要:
1. 遍历 `always_comb` 区域的所有操作
2. 识别外部引用的值
3. 创建等待块和分支结构

当存在嵌套模块和函数调用链时，观察值的收集可能变得复杂。

## 5. 最可能的根因

**主要根因**: 三级嵌套模块结构与 `always_comb` 内函数调用链的组合，导致:

1. **模块展开阶段**: 嵌套模块需要逐层展开，每层都创建独立的模块定义
2. **函数内联阶段**: `func1` → `func2` 的调用链需要在组合逻辑上下文中内联
3. **数据流分析阶段**: `data[0] = func1(data[7])` 的依赖关系需要精确分析
4. **Arc 转换阶段**: 将组合逻辑转换为 Arc 调用时的扇入分析

这些因素的组合可能导致某个 Pass 的算法复杂度呈指数增长，最终超时。

## 6. 建议

### 6.1 最小化方向
1. 移除一层嵌套模块，测试是否仍然超时
2. 移除函数调用链，直接使用表达式
3. 简化 `always_comb` 逻辑

### 6.2 调试建议
1. 使用 `-mlir-print-ir-after-all` 选项追踪 IR 变化
2. 使用 `-mlir-timing` 选项识别耗时的 Pass
3. 在各阶段添加超时检测

## 7. 相关源码位置

| 组件 | 文件路径 | 关键函数/类 |
|------|----------|-------------|
| Verilog 导入 | `lib/Conversion/ImportVerilog/Structure.cpp` | `convertModuleHeader`, `convertModuleBody` |
| Moore→Core | `lib/Conversion/MooreToCore/MooreToCore.cpp` | `ProcedureOpConversion` |
| Arc 转换 | `lib/Conversion/ConvertToArcs/ConvertToArcs.cpp` | `analyzeFanIn` |
| 循环分割 | `lib/Dialect/Arc/Transforms/SplitLoops.cpp` | `ensureNoLoops` |

## 8. 结论

这是一起由复杂代码结构触发的编译超时问题。嵌套模块、函数调用链和组合逻辑依赖的组合可能触发了 CIRCT 编译管道中某个分析 Pass 的性能问题。这不是传统的断言失败或段错误，而是算法复杂度问题导致的超时。
