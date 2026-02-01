# CIRCT 崩溃根因分析报告

## 基本信息

- **测试用例 ID**: 260129-00001624
- **崩溃类型**: Assertion Failure
- **工具**: circt-verilog (--ir-hw)
- **方言**: Moore → HW (MooreToCore 转换)

## 崩溃签名

```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

崩溃位置:
- 文件: `include/circt/Dialect/HW/PortImplementation.h:177`
- 函数: `circt::hw::ModulePortInfo::sanitizeInOut()`

## 根因分析

### 问题代码

测试用例定义了一个 **packed union** 类型作为模块输出端口：

```systemverilog
typedef union packed {
  logic [31:0] a;
  logic [31:0] b;
} my_union;

module top(output logic q, output logic ok, output my_union data);
  // ...
endmodule
```

### 崩溃原因

**MooreToCore pass 缺少对 `moore::UnionType` (packed union) 的类型转换支持。**

在 `lib/Conversion/MooreToCore/MooreToCore.cpp` 的类型转换器中：
- ✅ `StructType` → `hw::StructType` 有转换
- ✅ `UnpackedStructType` → `hw::StructType` 有转换
- ✅ `ArrayType` → `hw::ArrayType` 有转换
- ❌ **`UnionType` → ? 没有转换器**
- ❌ **`UnpackedUnionType` → ? 没有转换器**

### 崩溃流程

1. `circt-verilog --ir-hw` 解析 SystemVerilog 到 Moore dialect
2. MooreToCore pass 尝试转换 `moore.sv_module`
3. `getModulePortInfo()` (line 234) 调用 `typeConverter.convertType(port.type)` 处理端口类型
4. 对于 `my_union` 类型 (`moore::UnionType`)，没有注册的转换器
5. `convertType()` 返回空 `Type` (null type)
6. 构造 `hw::ModulePortInfo` 时，`sanitizeInOut()` 被调用
7. `sanitizeInOut()` 中 `dyn_cast<hw::InOutType>(p.type)` 对空类型操作
8. **断言失败**: `dyn_cast on a non-existent value`

### 关键代码位置

**缺失的类型转换** (`MooreToCore.cpp:2221-2341`):
```cpp
// 存在 StructType 转换
typeConverter.addConversion([&](StructType type) -> std::optional<Type> {
  // ...
  return hw::StructType::get(type.getContext(), fields);
});

// 缺少 UnionType 转换！
// typeConverter.addConversion([&](UnionType type) -> ...);
```

**崩溃点** (`PortImplementation.h:175-180`):
```cpp
void sanitizeInOut() {
  for (auto &p : ports)
    if (auto inout = dyn_cast<hw::InOutType>(p.type)) {  // p.type 是 null！
      p.type = inout.getElementType();
      p.dir = ModulePort::Direction::InOut;
    }
}
```

## 分类

| 属性 | 值 |
|------|-----|
| Bug 类型 | 缺失功能 / Missing Feature |
| 严重程度 | 中等 (崩溃但有明确错误输入) |
| 影响范围 | 使用 packed union 作为端口类型的设计 |
| 修复难度 | 中等 - 需要添加 Union 类型的转换逻辑 |

## 建议修复

在 `MooreToCore.cpp` 的类型转换器中添加对 `UnionType` 和 `UnpackedUnionType` 的处理。

**选项 A** - 转换为等宽整数类型（简单但丢失结构信息）：
```cpp
typeConverter.addConversion([&](UnionType type) -> std::optional<Type> {
  return IntegerType::get(type.getContext(), type.getBitSize());
});
```

**选项 B** - 转换为 struct（保留成员信息，但语义略有不同）：
```cpp
typeConverter.addConversion([&](UnionType type) -> std::optional<Type> {
  SmallVector<hw::StructType::FieldInfo> fields;
  for (auto field : type.getMembers()) {
    // ...
  }
  return hw::StructType::get(type.getContext(), fields);
});
```

**选项 C** - 在 HW dialect 中添加原生 Union 支持（最完整但工作量大）。

## 堆栈跟踪摘要

```
#17 circt::hw::ModulePortInfo::sanitizeInOut()     [PortImplementation.h:177]
#21 getModulePortInfo(TypeConverter, SVModuleOp)   [MooreToCore.cpp:259]
#22 SVModuleOpConversion::matchAndRewrite()        [MooreToCore.cpp:276]
#42 MooreToCorePass::runOnOperation()              [MooreToCore.cpp:2571]
```
