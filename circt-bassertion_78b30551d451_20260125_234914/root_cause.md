# Root Cause Analysis Report

## Executive Summary

circt-verilog 在处理带有 `string` 类型端口的模块时崩溃。当 Moore dialect 将 SystemVerilog 的 `string` 类型转换为 `sim::DynamicStringType` 后，`hw::ModulePortInfo` 的构造函数中的 `sanitizeInOut()` 方法尝试对该类型执行 `dyn_cast<hw::InOutType>`，但由于该类型不是有效的 HW dialect 类型，导致 `dyn_cast` 断言失败。

## Crash Context

- **Tool/Command**: `circt-verilog --ir-hw`
- **Dialect**: Moore → HW/Comb/LLHD
- **Failing Pass**: MooreToCore (SVModuleOpConversion)
- **Crash Type**: Assertion failure

## Error Analysis

### Assertion/Error Message

```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

### Key Stack Frames

```
#11 0x000055c8ff7cbb57 (dyn_cast<InOutType>)
#12 0x000055c8ffdd4717 getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp)
                       MooreToCore.cpp:259
#13 0x000055c8ffdd4717 SVModuleOpConversion::matchAndRewrite(...)
                       MooreToCore.cpp:276
```

### Crash Location Analysis

崩溃发生在 `getModulePortInfo()` 函数返回时（MooreToCore.cpp:259）。该函数创建的 `hw::ModulePortInfo` 对象在构造时调用 `sanitizeInOut()` 方法，该方法对所有端口类型执行 `dyn_cast<hw::InOutType>`。

## Test Case Analysis

### Code Summary

```systemverilog
module top(input string a, output logic [7:0] out);
  int length;
  logic [7:0] in = 8'hFF;

  always_comb begin
    length = a.len();
  end
  
  assign out[7-:4] = in;
endmodule
```

测例定义了一个模块，其输入端口 `a` 的类型为 `string`（SystemVerilog 动态字符串类型）。

### Key Constructs

- **`input string a`**: 动态字符串类型作为模块端口 — **这是导致崩溃的根本原因**
- `a.len()`: 字符串内置方法调用
- `out[7-:4] = in`: 部分选择赋值（与崩溃无关）

### Potentially Problematic Patterns

1. **String type as module port**: SystemVerilog 的 `string` 是动态类型，在硬件综合语义中没有明确定义
2. 转换后的 `sim::DynamicStringType` 不是 HW dialect 支持的端口类型

## CIRCT Source Analysis

### Crash Location

**File**: `lib/Conversion/MooreToCore/MooreToCore.cpp`  
**Function**: `getModulePortInfo()`  
**Line**: ~259 (return statement)

### Processing Path

```
1. SVModuleOpConversion::matchAndRewrite() [line 276]
   ↓ 调用
2. getModulePortInfo(typeConverter, op) [line 276]
   ↓ 遍历端口
3. typeConverter.convertType(port.type) [line 243]
   - StringType → sim::DynamicStringType ✓
   ↓ 构造 PortInfo
4. hw::PortInfo({{port.name, portTy, port.dir}, inputNum++, {}}) [line 253-254]
   ↓ 返回
5. return hw::ModulePortInfo(ports) [line 258]
   ↓ 构造函数
6. ModulePortInfo 构造函数调用 sanitizeInOut() [PortImplementation.h:67]
   ↓
7. dyn_cast<hw::InOutType>(p.type) [PortImplementation.h:177]
   ↓ 类型不是 HW dialect 的有效类型
8. ASSERTION FAILED: "dyn_cast on a non-existent value"
```

### Code Context

**MooreToCore.cpp:233-259** (getModulePortInfo):
```cpp
static hw::ModulePortInfo getModulePortInfo(const TypeConverter &typeConverter,
                                            SVModuleOp op) {
  ...
  for (auto port : moduleTy.getPorts()) {
    Type portTy = typeConverter.convertType(port.type);  // StringType → sim::DynamicStringType
    if (port.dir == hw::ModulePort::Direction::Output) {
      ports.push_back(hw::PortInfo(...));
    } else {
      ports.push_back(hw::PortInfo({{port.name, portTy, port.dir}, inputNum++, {}}));
    }
  }
  return hw::ModulePortInfo(ports);  // <-- 这里触发 sanitizeInOut()
}
```

**PortImplementation.h:175-180** (sanitizeInOut):
```cpp
void sanitizeInOut() {
  for (auto &p : ports)
    if (auto inout = dyn_cast<hw::InOutType>(p.type)) {  // <-- CRASH HERE
      p.type = inout.getElementType();
      p.dir = ModulePort::Direction::InOut;
    }
}
```

### Type Conversion Chain

```
moore::StringType (SystemVerilog string)
    ↓ typeConverter.addConversion [MooreToCore.cpp:2277-2278]
sim::DynamicStringType
    ↓ 用作 hw::PortInfo.type
    ↓ dyn_cast<hw::InOutType>
ASSERTION FAILURE (不是有效的 HW type)
```

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)

**Cause**: `hw::ModulePortInfo::sanitizeInOut()` 假设所有端口类型都是 HW dialect 兼容类型，但 `sim::DynamicStringType` 不满足这个假设。

**Evidence**:
1. `dyn_cast<hw::InOutType>` 只能安全地用于 HW dialect 类型
2. `sim::DynamicStringType` 是 Sim dialect 类型，不属于 HW 类型系统
3. Stack trace 明确指向 `dyn_cast` 调用失败

**Mechanism**:
`dyn_cast` 在 MLIR 中对于不兼容的类型会返回 null，但在某些 LLVM/MLIR 配置下，当类型完全不匹配时会触发 `isPresent` 断言失败。`sim::DynamicStringType` 既不是 `hw::InOutType`，也不是 HW dialect 的任何已知类型，这导致了断言失败。

### Hypothesis 2 (Medium Confidence)

**Cause**: MooreToCore 转换中缺少对非综合类型（如 `string`）作为端口类型的验证和错误处理。

**Evidence**:
1. `getModulePortInfo` 没有验证转换后的 `portTy` 是否是有效的 HW 端口类型
2. SystemVerilog `string` 类型在硬件综合中通常不被支持

**Mechanism**:
应该在转换前检查端口类型是否可综合，对于 `string` 等仿真专用类型，应该提前报错而不是尝试转换。

### Hypothesis 3 (Lower Confidence)

**Cause**: `sanitizeInOut()` 的实现过于激进，应该只对可能是 `InOutType` 的类型执行检查。

**Evidence**:
1. `dyn_cast` 应该对不兼容类型返回 null 而不是崩溃
2. 可能是 MLIR 类型系统在特定配置下的行为问题

## Suggested Fix Directions

### Option 1: 在 getModulePortInfo 中验证端口类型（推荐）

```cpp
Type portTy = typeConverter.convertType(port.type);
if (!portTy || !hw::isHWValueType(portTy)) {
  // 发出诊断错误并返回 failure
  return emitError(op.getLoc()) << "unsupported port type: " << port.type;
}
```

### Option 2: 在 sanitizeInOut 中增加类型检查

```cpp
void sanitizeInOut() {
  for (auto &p : ports) {
    if (!p.type)
      continue;
    if (auto inout = dyn_cast_if_present<hw::InOutType>(p.type)) {
      p.type = inout.getElementType();
      p.dir = ModulePort::Direction::InOut;
    }
  }
}
```

### Option 3: 在类型转换中标记不可综合类型

为 `sim::DynamicStringType` 等仿真类型添加验证，在用于端口时报告明确的错误信息。

## Keywords for Issue Search

`string` `StringType` `DynamicStringType` `port` `getModulePortInfo` `sanitizeInOut` `InOutType` `dyn_cast` `MooreToCore` `assertion`

## Related Files to Investigate

- `lib/Conversion/MooreToCore/MooreToCore.cpp` - 转换逻辑
- `include/circt/Dialect/HW/PortImplementation.h` - `sanitizeInOut` 实现
- `include/circt/Dialect/HW/HWTypes.h` - HW 类型定义
- `include/circt/Dialect/Moore/MooreTypes.h` - Moore StringType 定义
