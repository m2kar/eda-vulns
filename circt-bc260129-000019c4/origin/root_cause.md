# CIRCT Crash Root Cause Analysis

## Crash ID: 260129-000019c4

## 1. 错误上下文

### 错误类型
Assertion failure

### 错误消息
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
```

### 崩溃位置
- **文件**: `circt/Dialect/Arc/Transforms/LowerState.cpp:219`
- **函数**: `circt::arc::StateType::get()`
- **调用栈**:
  1. `LowerStatePass::runOnOperation()` (LowerState.cpp:1198)
  2. `ModuleLowering::run()` (LowerState.cpp:219)
  3. `StateType::get(mlir::Type)` (ArcTypes.cpp.inc:108)

### 复现命令
```bash
circt-verilog --ir-hw source.sv | arcilator
```

## 2. 测例分析

### 源代码 (source.sv)
```systemverilog
module top_module (
  inout wire my_pin       // <-- 关键：inout 端口
);

  logic [7:0] data_array [0:3];
  
  initial begin
    data_array <= '{default: 8'hFF};
  end
  
  assign my_pin = data_array[0][0] ? 1'bz : 1'b0;  // <-- 三态逻辑

endmodule
```

### 关键构造分析
| 特性 | 说明 |
|------|------|
| `inout wire` 端口 | 双向端口，触发 `llhd::RefType` 生成 |
| 三态赋值 (`1'bz`) | 高阻态输出 |
| 条件三态表达式 | `data_array[0][0] ? 1'bz : 1'b0` |
| 多维数组索引 | `data_array[0][0]` |

## 3. 源码定位与分析

### 3.1 崩溃点: LowerState.cpp:219

```cpp
// Allocate storage for the inputs.
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(arg.getType()), name, storageArg);  // <-- 第219行
  allocatedInputs.push_back(state);
}
```

**问题**: 此代码遍历模块的所有 block arguments（输入端口），并尝试为每个参数创建 `StateType`。当参数类型为 `!llhd.ref<i1>`（来自 inout 端口）时，`StateType::get()` 验证失败。

### 3.2 StateType 验证: ArcTypes.cpp:80-87

```cpp
LogicalResult
StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  Type innerType) {
  if (!computeLLVMBitWidth(innerType))  // <-- 验证是否有已知位宽
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

**问题**: `computeLLVMBitWidth()` 函数只处理以下类型：
- `seq::ClockType`
- `IntegerType`
- `hw::ArrayType`
- `hw::StructType`

它**不支持** `llhd::RefType`，因此对 `!llhd.ref<i1>` 返回 `std::nullopt`，导致验证失败。

### 3.3 类型转换路径

inout 端口的类型转换流程：
1. **Slang 解析**: SystemVerilog inout 端口
2. **Moore 方言**: `moore.ref<...>` 类型
3. **MooreToCore 转换**: `moore.ref<T>` → `llhd.ref<T>` (MooreToCore.cpp:2304-2308)
   ```cpp
   typeConverter.addConversion([&](RefType type) -> std::optional<Type> {
     if (auto innerType = typeConverter.convertType(type.getNestedType()))
       return llhd::RefType::get(innerType);
     return {};
   });
   ```
4. **HWModuleOp**: Block argument 类型为 `!llhd.ref<i1>`
5. **LowerState**: 尝试创建 `arc.state<!llhd.ref<i1>>` → **崩溃**

## 4. 根因假设

### 主要根因
**Arc 方言的 LowerState pass 不支持 `llhd::RefType` 类型作为模块输入**

具体来说：
1. `inout` 端口在 MooreToCore 转换后变成 `!llhd.ref<T>` 类型的 block argument
2. LowerState pass 假设所有输入端口类型都有可计算的 bit width
3. `llhd::RefType` 是一种引用/指针类型，不应该有固定的 bit width
4. `StateType::verify()` 对此类型验证失败

### 根本问题
Arc 方言（用于模拟器生成）目前不支持双向端口（inout）的语义：
- 引用类型（`llhd::RefType`）表示可以被读写的信号引用
- 模拟器存储（`arc.state`）需要固定大小的存储空间
- 两者语义不兼容

### 相关代码注释
MooreToCore.cpp 中有明确的 FIXME 注释说明此问题尚未完全解决：
```cpp
// FIXME: Once we support net<...>, ref<...> type to represent type of
// special port like inout or ref port which is not a input or output
// port. It can change to generate corresponding types for direction of
// port or do specified operation to it. Now inout and ref port is treated
// as input port.
```

## 5. 问题分类

| 属性 | 值 |
|------|-----|
| Bug 类型 | Missing Feature / Assertion Failure |
| 方言 | arc |
| 影响范围 | arcilator (模拟器) |
| 触发条件 | 含 inout 端口的模块 |
| 严重程度 | 高（导致工具崩溃） |

## 6. 建议修复方向

### 短期方案
在 `LowerState::run()` 中添加对 `llhd::RefType` 的检测，跳过或报告不支持：

```cpp
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  if (isa<llhd::RefType>(arg.getType())) {
    return moduleOp.emitError() 
        << "inout ports are not supported by arcilator";
  }
  // ... 现有代码
}
```

### 长期方案
1. 扩展 Arc 方言以支持引用类型的模拟语义
2. 在 arcilator 前端或管道早期阶段拒绝不支持的输入
3. 实现 inout 端口到可模拟表示的转换

## 7. 关键词

- `inout port`
- `llhd.ref`
- `StateType`
- `LowerState`
- `arcilator`
- `bidirectional port`
- `tri-state logic`
