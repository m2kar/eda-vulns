# MooreToCore UnionType 转换调研报告

**调研日期**: 2026-01-31
**调研范围**: CIRCT 项目中 MooreToCore 转换对 UnionType 的支持情况
**项目版本**: fix-b3 分支 (commit e4838c703)

---

## 执行摘要

本次调研发现 **MooreToCore 转换缺少 UnionType 支持**，这是一个已知的功能缺口。虽然 UnionType 在 Moore dialect、ImportVerilog、ExportVerilog 和 Python bindings 中都有完整支持，但在 MooreToCore 转换 pass 中尚未实现类型转换器和操作转换模式。

---

## 1. 核心发现

### 1.1 MooreToCore 缺少 UnionType 转换支持

**代码位置**: [lib/Conversion/MooreToCore/MooreToCore.cpp:2283-2408](../lib/Conversion/MooreToCore/MooreToCore.cpp#L2283-L2408)

**已实现的类型转换**:
```cpp
// StructType 转换 (lines 2324-2335)
typeConverter.addConversion([&](StructType type) -> std::optional<Type> {
  SmallVector<hw::StructType::FieldInfo> fields;
  for (auto field : type.getMembers()) {
    hw::StructType::FieldInfo info;
    info.type = typeConverter.convertType(field.type);
    if (!info.type)
      return {};
    info.name = field.name;
    fields.push_back(info);
  }
  return hw::StructType::get(type.getContext(), fields);
});

// UnpackedStructType 转换 (lines 2342-2354)
typeConverter.addConversion([&](UnpackedStructType type) -> std::optional<Type> {
  // Similar implementation
});
```

**缺失的转换**:
- ❌ `moore::UnionType` → `hw::StructType` (或其他目标类型)
- ❌ `moore::UnpackedUnionType` → 目标类型
- ❌ Union 相关操作的转换模式 (extract, create, inject)

---

## 2. UnionType 在 Moore Dialect 中的定义

### 2.1 类型定义

**文件**: [include/circt/Dialect/Moore/MooreTypes.td:384-406](../include/circt/Dialect/Moore/MooreTypes.td#L384-L406)

```tablegen
def UnionType : StructLikeType<"Union", [
  DeclareTypeInterfaceMethods<DestructurableTypeInterface>
], "moore::PackedType"> {
  let mnemonic = "union";
  let summary = "a packed union type";
  let description = [{
    A packed union. All members are guaranteed to be packed as well.
  }];
  let genVerifyDecl = 1;
}

def UnpackedUnionType : StructLikeType<"UnpackedUnion", [
  DeclareTypeInterfaceMethods<DestructurableTypeInterface>
], "moore::UnpackedType"> {
  let mnemonic = "uunion";
  let summary = "an unpacked union type";
  let description = [{
    An unpacked union. Members can be packed or unpacked.
  }];
  let genVerifyDecl = 1;
}
```

### 2.2 类型验证

**文件**: [lib/Dialect/Moore/MooreTypes.cpp:245-248](../lib/Dialect/Moore/MooreTypes.cpp#L245-L248)

```cpp
LogicalResult UnionType::verify(function_ref<InFlightDiagnostic()> emitError,
                                ArrayRef<StructLikeMember> members) {
  return verifyAllMembersPacked(emitError, members);
}
```

**约束**: 所有 UnionType 成员必须是 packed types

### 2.3 接口实现

UnionType 实现了 `DestructurableTypeInterface`:
- `getSubelementIndexMap()` (line 309)
- `getTypeAtIndex()` (line 313)
- `getFieldIndex()` (line 317)

---

## 3. HW Dialect 中的 Union 语义

### 3.1 布局规范

**文档**: [docs/Dialects/Comb/RationaleComb.md:302-306](../docs/Dialects/Comb/RationaleComb.md#L302-L306)

```
- Unions: The HW dialect's UnionType could contain the data of any of the
  member types so its layout is defined to be equivalent to the union of members
  type bitcast layout. In cases where the member types have different bit widths,
  all members start at the 0th bit and are padded up to the width of the widest
  member. The value with which they are padded is undefined.
```

**关键设计点**:
1. Union 可以包含任意成员类型的数据
2. 所有成员从第 0 位开始（重叠布局）
3. 宽度不同时填充到最宽成员的宽度
4. 填充值未定义 (undefined)

### 3.2 HW Rationale 说明

**文档**: [docs/Dialects/HW/RationaleHW.md:89-92](../docs/Dialects/HW/RationaleHW.md#L89-L92)

```
### `union` Type

Union types contain a single data element (which may be an aggregate).
They optionally have an offset per variant which allows non-SV layouts.
```

---

## 4. 相关 GitHub Issues 和 PRs

### 4.1 Issue #8471: Union type in call

**URL**: https://github.com/llvm/circt/issues/8471
**状态**: OPEN
**问题描述**: 函数调用时 union type 的类型不匹配

```
error: 'func.call' op operand type mismatch:
expected operand type '!moore.union<..., A, ...>', but provided 'A'
```

**根本原因**: Slang 的类型检查可能没有为 union 插入显式转换 AST 节点

**讨论要点** (fabianschuiki):
> Usually, Slang's type checking inserts explicit conversion AST nodes where necessary.
> It might not do this for unions though, since there's technically no conversion happening.

**可能解决方案**: 升级到 Slang v8 可能会解决此问题

### 4.2 相关 PRs

- **PR #7341**: [Moore] [Canonicalizer] Lower struct-related assignOp (OPEN)
  - 处理 struct 相关的赋值操作
  - 可能为 union 提供参考实现

---

## 5. Git 历史中的 Union 支持演进

### 5.1 关键提交记录

```bash
7a0ad42ae [HW] Extend ElementType parsing to support union types (#9318)
bf9706713 [ExportVerilog] Fix union emission error
97814a7ad [PyCDE] Add support for union types (#9242)
431b1ac9d [Python] Add support for UnionType (#9236)
00edb48ed [ImportVerilog][Moore] Support union type (#7084)
b8d69d5cc [HW] Reference struct/union fields by index (#6266)
a4b8f96f5 [ExportVerilog] Add support for UnionCreate op. (#5081)
```

### 5.2 时间线分析

| 时间 | 功能 | 状态 |
|------|------|------|
| 2023 | HW dialect 添加 union 基础支持 | ✅ 完成 |
| 2024 | ImportVerilog 和 Moore dialect 添加 union 支持 | ✅ 完成 |
| 2024 | Python bindings 和 PyCDE 添加 union 支持 | ✅ 完成 |
| 2024 | ExportVerilog 支持 union 导出 | ✅ 完成 |
| **至今** | **MooreToCore 转换** | ❌ **未实现** |

---

## 6. 测试覆盖情况

### 6.1 现有测试

**UnionType 验证测试**:
- **文件**: [test/Dialect/Moore/types-errors.mlir:19](../test/Dialect/Moore/types-errors.mlir#L19)
```mlir
// expected-error @below {{StructType/UnionType members must be packed types}}
unrealized_conversion_cast to !moore.union<{foo: string}>
```

**StructType 转换测试**:
- **文件**: [test/Conversion/MooreToCore/basic.mlir:600-623](../test/Conversion/MooreToCore/basic.mlir#L600-L623)
- 包含 `moore.struct_extract`, `moore.struct_create` 等操作的转换测试

### 6.2 缺失测试

- ❌ MooreToCore 中没有 UnionType 转换测试
- ❌ 没有 `moore.union_extract` 转换测试
- ❌ 没有 `moore.union_create` 转换测试
- ❌ 没有 `moore.union_inject` 转换测试

---

## 7. 其他 MooreToCore 已知问题

相关的 MooreToCore 转换问题（供参考）:

| Issue | 标题 | 状态 |
|-------|------|------|
| #9542 | to_builtin_bool should be replaced with to_builtin_int | OPEN |
| #8973 | Lowering to math.ipow? | OPEN |
| #8930 | Crash with sqrt/floor | OPEN |
| #8269 | Support `real` constants | OPEN |
| #7629 | Support net op | OPEN |
| #8276 | Support for UnpackedArrayType emission | OPEN |
| #8332 | Support for StringType from moore to llvm dialect | OPEN |

---

## 8. 实现建议

### 8.1 UnionType 类型转换器

基于现有的 StructType 转换模式，建议实现：

```cpp
// 添加到 MooreToCore.cpp 的 type converter setup 中
typeConverter.addConversion([&](UnionType type) -> std::optional<Type> {
  SmallVector<hw::StructType::FieldInfo> fields;
  for (auto field : type.getMembers()) {
    hw::StructType::FieldInfo info;
    info.type = typeConverter.convertType(field.type);
    if (!info.type)
      return {};
    info.name = field.name;
    fields.push_back(info);
  }
  // Union 在硬件中表示为 struct（字段在内存中重叠）
  // HW dialect 的 UnionType 或 StructType 都可以作为目标
  return hw::StructType::get(type.getContext(), fields);
});

typeConverter.addConversion([&](UnpackedUnionType type) -> std::optional<Type> {
  SmallVector<hw::StructType::FieldInfo> fields;
  for (auto field : type.getMembers()) {
    hw::StructType::FieldInfo info;
    info.type = typeConverter.convertType(field.type);
    if (!info.type)
      return {};
    info.name = field.name;
    fields.push_back(info);
  }
  return hw::StructType::get(type.getContext(), fields);
});
```

### 8.2 Union 操作转换模式

需要添加以下操作的转换模式：

1. **UnionExtractOp**: 从 union 中提取字段
   - 参考 `StructExtractOpConversion` 实现
   - 可能需要 bitcast 操作

2. **UnionCreateOp**: 创建 union 值
   - 参考 `StructCreateOpConversion` 实现
   - 需要处理字段重叠的语义

3. **UnionInjectOp**: 向 union 注入值
   - 类似于 struct inject 的实现

### 8.3 测试用例

建议在 `test/Conversion/MooreToCore/basic.mlir` 中添加：

```mlir
// Union type conversion test
func.func @union_ops(%arg0: !moore.union<{a: i32, b: i16}>) {
  // Test union extract
  %0 = moore.union_extract %arg0, "a" : !moore.union<{a: i32, b: i16}> -> i32

  // Test union create
  %1 = moore.constant 42 : i32
  %2 = moore.union_create "a", %1 : i32 -> !moore.union<{a: i32, b: i16}>

  return
}
```

---

## 9. 结论

### 9.1 UnionType 支持现状总结

| 组件 | 支持状态 | 说明 |
|------|---------|------|
| Moore Dialect 定义 | ✅ 完整支持 | 类型定义、验证、接口实现完整 |
| ImportVerilog | ✅ 完整支持 | 可以从 SystemVerilog 导入 union |
| ExportVerilog | ✅ 完整支持 | 可以导出为 SystemVerilog union |
| Python Bindings | ✅ 完整支持 | Python API 可用 |
| PyCDE | ✅ 完整支持 | 高层前端支持 |
| **MooreToCore 转换** | ❌ **未实现** | **功能缺口** |
| HW Dialect 语义 | ✅ 文档完整 | 布局规范清晰 |

### 9.2 关键结论

1. **这是一个已知的未实现功能**，而不是设计上的限制
2. **实现路径清晰**，可以参考现有的 StructType 转换模式
3. **优先级可能不高**，因为相关 issue (#8471) 仍处于 OPEN 状态
4. **实现难度中等**，主要工作量在于：
   - 添加类型转换器（相对简单）
   - 实现操作转换模式（需要处理字段重叠语义）
   - 编写完整的测试用例

### 9.3 与 Bug B3 的关系

本次调研是为了理解 Bug B3 (UnionType 转换崩溃) 的背景。调研结果表明：

- **根本原因**: MooreToCore 根本没有实现 UnionType 的转换支持
- **Bug 表现**: 当遇到 UnionType 时，类型转换器返回 null，导致后续代码崩溃
- **修复方向**: 需要完整实现 UnionType 的转换支持，而不仅仅是修复崩溃

---

## 10. 参考资料

### 10.1 代码文件

- [lib/Conversion/MooreToCore/MooreToCore.cpp](../lib/Conversion/MooreToCore/MooreToCore.cpp)
- [include/circt/Dialect/Moore/MooreTypes.td](../include/circt/Dialect/Moore/MooreTypes.td)
- [lib/Dialect/Moore/MooreTypes.cpp](../lib/Dialect/Moore/MooreTypes.cpp)
- [test/Conversion/MooreToCore/basic.mlir](../test/Conversion/MooreToCore/basic.mlir)

### 10.2 文档

- [docs/Dialects/HW/RationaleHW.md](../docs/Dialects/HW/RationaleHW.md)
- [docs/Dialects/Comb/RationaleComb.md](../docs/Dialects/Comb/RationaleComb.md)

### 10.3 GitHub Issues

- Issue #8471: https://github.com/llvm/circt/issues/8471
- PR #7341: https://github.com/llvm/circt/pull/7341

---

**报告结束**
