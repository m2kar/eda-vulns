# CIRCT 崩溃根因分析报告

## 崩溃概述

| 属性 | 值 |
|------|-----|
| **崩溃类型** | Assertion Failure |
| **错误信息** | `integer bitwidth is limited to 16777215 bits` |
| **崩溃位置** | `Mem2Reg.cpp:1742, insertBlockArgs` |
| **方言** | LLHD (通过 Moore 方言处理 SystemVerilog) |
| **工具** | `circt-verilog` |
| **Testcase ID** | 260129-0000175a |

## 崩溃调用栈分析

```
#11 IntegerType::get() 验证失败
#12 mlir::IntegerType::get(mlir::MLIRContext*, unsigned int, ...)
#13 (anonymous namespace)::Promoter::insertBlockArgs((anonymous namespace)::BlockEntry*)
    -> Mem2Reg.cpp:1742
#14 (anonymous namespace)::Promoter::insertBlockArgs()
    -> Mem2Reg.cpp:1654
#15 (anonymous namespace)::Promoter::promote()
    -> Mem2Reg.cpp:764
#16 Mem2RegPass::runOnOperation()
    -> Mem2Reg.cpp:1844
```

崩溃发生在 LLHD 方言的 Mem2Reg (内存到寄存器提升) Pass 中，具体是在 `insertBlockArgs` 函数内部创建 `IntegerType` 时，因为位宽超过了 MLIR IntegerType 的最大限制 (16,777,215 bits = 0xFFFFFF)。

## 测例分析

### 测例代码

```systemverilog
module top_module(input logic clk, input logic resetn);
  
  // Class definition with typedef referencing itself in parameterized template
  class registry #(type T = int);
    // Parameterized registry class
  endclass
  
  class my_class;
    typedef registry#(my_class) type_id;  // <- 循环引用
  endclass
  
  // Instance of the class
  my_class obj;
  
  // Sequential logic with clock edge
  always_ff @(posedge clk) begin
    if (resetn) begin
      obj = new();
    end
  end
  
endmodule
```

### 关键构造

1. **参数化类模板**: `registry #(type T = int)` - 一个接受类型参数的泛型类
2. **循环类型引用**: `class my_class` 内部定义 `typedef registry#(my_class) type_id`
3. **类实例化**: `my_class obj` 后在 `always_ff` 块中调用 `obj = new()`

## 根因分析

### 直接原因

在 `Mem2Reg.cpp:1753` 行:

```cpp
auto flatType = builder.getIntegerType(hw::getBitWidth(type));
```

当 `getStoredType(slot)` 返回的类型（可能是 `ClassHandleType` 或相关类型）传入 `hw::getBitWidth()` 时，返回了一个极大的值（超过 16,777,215），导致 `IntegerType::get()` 的验证失败。

### 根本原因假设

#### 假设 1: ClassHandleType 位宽计算问题 (高可能性)

`ClassHandleType` 是一个指向堆上类对象的句柄类型。当 Mem2Reg Pass 试图将包含类句柄的信号提升为寄存器时：

1. `getStoredType()` 返回 `ClassHandleType` 或其相关引用类型
2. `hw::getBitWidth()` 对于 `ClassHandleType` 可能返回异常值（-1 被错误转换，或者类型循环导致溢出）
3. 使用这个异常的位宽值创建 `IntegerType` 时触发断言

#### 假设 2: 循环类型引用导致计算溢出 (中等可能性)

`my_class` 内部的 `typedef registry#(my_class) type_id` 形成了类型的自引用结构：

- `my_class` -> `registry#(my_class)` -> `my_class` ...

如果位宽计算过程中没有正确检测和处理这种循环引用，可能导致：
- 无限递归（被其他机制终止后产生垃圾值）
- 整数溢出
- 返回未初始化的值

#### 假设 3: 类类型不应参与 Mem2Reg 提升 (高可能性)

SystemVerilog 的类对象是堆分配的引用类型，其 "位宽" 概念与普通硬件类型不同：

1. `hw::getBitWidth()` 设计时可能未考虑 `ClassHandleType`
2. 对于不支持的类型，`getBitWidth()` 返回 -1
3. 当 -1 (作为 int64_t) 被转换为 unsigned 传给 `getIntegerType()` 时，变成极大正数

### 源码验证

#### hw::getBitWidth() 实现 (HWTypes.cpp:110)

```cpp
int64_t circt::hw::getBitWidth(mlir::Type type) {
  return llvm::TypeSwitch<::mlir::Type, int64_t>(type)
      .Case<IntegerType>(
          [](IntegerType t) { return t.getIntOrFloatBitWidth(); })
      .Default([](Type type) -> int64_t {
        // If type implements BitWidthTypeInterface, use it.
        if (auto iface = dyn_cast<BitWidthTypeInterface>(type)) {
          std::optional<int64_t> width = iface.getBitWidth();
          return width.has_value() ? *width : -1;
        }
        return -1;  // <- ClassHandleType 可能落入这里返回 -1
      });
}
```

#### Mem2Reg.cpp:1752-1753 崩溃点

```cpp
auto type = getStoredType(slot);
auto flatType = builder.getIntegerType(hw::getBitWidth(type));
//                                     ^^^^^^^^^^^^^^^^^^^^^^^^^
//                        如果 type 是 ClassHandleType, 这里返回 -1
//                        转换为 unsigned 后变成 0xFFFFFFFFFFFFFFFF (极大值)
```

### 结论

**最可能的根因**: LLHD Mem2Reg Pass 在处理包含类对象引用 (`ClassHandleType`) 的信号时，没有正确检查类型是否支持位宽计算。当 `hw::getBitWidth()` 返回 -1（表示不支持或未知类型）时，这个负值被隐式转换为极大的无符号整数，随后传递给 `IntegerType::get()` 导致断言失败。

## 影响组件

1. **LLHD Dialect - Mem2Reg Pass** (`lib/Dialect/LLHD/Transforms/Mem2Reg.cpp`)
   - 需要在创建 IntegerType 前验证 getBitWidth() 返回值
   - 应过滤不支持位宽计算的类型（如 ClassHandleType）

2. **HW Dialect - Type System** (`lib/Dialect/HW/HWTypes.cpp`)
   - `getBitWidth()` 对于 ClassHandleType 返回 -1
   - 调用方需要正确处理这个返回值

3. **Moore Dialect - Class Types**
   - `ClassHandleType` 作为句柄类型，其 "位宽" 语义不明确
   - 可能需要实现 `BitWidthTypeInterface` 并返回指针大小

## 修复建议

### 短期修复

在 `Mem2Reg.cpp` 中的 `insertBlockArgs()` 函数添加类型检查：

```cpp
auto type = getStoredType(slot);
int64_t bitWidth = hw::getBitWidth(type);
if (bitWidth < 0) {
  // Handle unsupported types gracefully
  return emitError(...) << "unsupported type for mem2reg promotion: " << type;
}
auto flatType = builder.getIntegerType(static_cast<unsigned>(bitWidth));
```

### 长期修复

1. 在 Mem2Reg Pass 的 `findPromotableSlots()` 阶段过滤掉包含 `ClassHandleType` 的 slot
2. 为 `ClassHandleType` 定义明确的内存表示（如固定大小的指针）
3. 添加编译器诊断，当类类型被用于不支持的硬件上下文时给出清晰错误

## 复现命令

```bash
circt-verilog --ir-hw testcase.sv
```

## 相关 Issues

- 需要检查是否有相关的 CIRCT Issue 报告
- SystemVerilog 类支持目前处于实验阶段（见编译警告）
