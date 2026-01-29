# [MooreToCore] Assertion failure when using string type as module output port

## Bug Report

### Description

`circt-verilog` crashes with an assertion failure when a SystemVerilog module declares a `string` type as an output port. The crash occurs in the MooreToCore conversion pass in `getModulePortInfo()`.

### Minimal Reproducer

```systemverilog
module test(output string result);
endmodule
```

### Crash Command

```bash
circt-verilog --ir-hw test.sv
```

### Error Output

```
circt-verilog: llvm/llvm/include/llvm/Support/Casting.h:650: decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]: Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
Stack dump:
0.	Program arguments: circt-verilog --ir-hw test.sv
 #11	(circt-verilog+0x1dbcb57)
 #12	llvm::SmallVector<circt::hw::PortInfo, 1u>::~SmallVector()
 #13	(anonymous namespace)::getModulePortInfo(...) MooreToCore.cpp:259
 #14	(anonymous namespace)::SVModuleOpConversion::matchAndRewrite(...) MooreToCore.cpp:276
 #35	(anonymous namespace)::MooreToCorePass::runOnOperation() MooreToCore.cpp:2571
...
Aborted (core dumped)
```

### Version Information

- **CIRCT Version**: firtool-1.139.0
- **LLVM Version**: 22.0.0git
- **Crash Hash**: dac29ed967fa

## Root Cause Analysis

### Crash Location

- **File**: `lib/Conversion/MooreToCore/MooreToCore.cpp`
- **Function**: `getModulePortInfo(const TypeConverter &typeConverter, SVModuleOp op)`
- **Line**: 259 (during `SmallVector<hw::PortInfo>` destruction)

### Root Cause

The crash occurs due to a **missing validation for converted port types**:

1. The `moore::StringType` port is converted to `sim::DynamicStringType` (line 2277-2279)
2. This `sim::DynamicStringType` is used to construct `hw::PortInfo` (line 243, 89-96)
3. `sim::DynamicStringType` is a simulation-only type, not valid for hardware ports
4. During `SmallVector<hw::PortInfo>` destruction, `dyn_cast<hw::InOutType>` is called on the invalid type
5. The assertion fails because the type is not present for casting

### Type Conversion Chain

```cpp
// lib/Conversion/MooreToCore/MooreToCore.cpp:2277-2279
typeConverter.addConversion([&](StringType type) {
  return sim::DynamicStringType::get(type.getContext());
});

// lib/Conversion/MooreToCore/MooreToCore.cpp:243
Type portTy = typeConverter.convertType(port.type);  // Returns sim::DynamicStringType
ports.push_back(hw::PortInfo({{port.name, portTy, port.dir}, resultNum++, {}}));
```

### The Gap

The code at line 243 does not validate if the converted type is valid for hardware ports before constructing `hw::PortInfo`. While `sim::DynamicStringType` is a valid MLIR type, it's not suitable for hardware module ports.

## Validation

### Syntax Validity

| Aspect | Result | Notes |
|--------|--------|-------|
| **Language** | SystemVerilog | IEEE 1800-2017 |
| **Syntax Valid** | ✅ Yes | Confirmed by Slang |
| **IEEE Compliant** | ✅ Yes | String ports are legal per IEEE 1800 Section 6.16 |

### Cross-Tool Validation

| Tool | Result | Output |
|------|--------|--------|
| **Slang** | ✅ PASSED | `Build succeeded: 0 errors, 0 warnings` |
| **Verilator** | ✅ PASSED | Exit code 0 (cosmetic warnings only) |
| **Icarus** | ❌ FAILED | `Port 'result' with type 'string' is not supported` |
| **CIRCT** | 💥 CRASH | Assertion failure |

**Conclusion**: The test case uses valid SystemVerilog syntax. Even if `string` ports are not synthesizable, CIRCT should emit a diagnostic error, not crash.

### Classification

- **Type**: `valid_testcase` - Genuine bug
- **Severity**: High (crash on valid input)
- **Bug Type**: `crash_on_valid_input`

## Suggested Fix

### Recommended Approach: Add validation in `getModulePortInfo()`

```cpp
static FailureOr<hw::ModulePortInfo> getModulePortInfo(const TypeConverter &typeConverter,
                                                       SVModuleOp op) {
  // ... existing code ...
  for (auto port : moduleTy.getPorts()) {
    Type portTy = typeConverter.convertType(port.type);

    // ADD VALIDATION HERE
    if (!portTy || !hw::isHWValueType(portTy)) {
      op.emitError() << "port '" << port.name
                    << "' has unsupported type for hardware: "
                    << port.type;
      return failure();
    }

    // ... rest of existing code ...
  }
  return hw::ModulePortInfo(ports);
}
```

### Alternative: Early Rejection in Moore Dialect

Add a verifier in the Moore dialect to reject `string` type ports with a clear error message like:
```
string type ports are not supported for hardware synthesis
```

## Related Issues

⚠️ **This bug is related to existing issues about string type support in MooreToCore:**

| Issue | Title | Score | Relevance |
|-------|-------|-------|-----------|
| #8283 | [ImportVerilog] Cannot compile forward declared string type | 12.5 | Directly reports MooreToCore lacks string-type conversion |
| #8332 | [MooreToCore] Support for StringType from moore to llvm dialect | 11.0 | Feature request for StringType support |
| #8930 | [MooreToCore] Crash with sqrt/floor | 9.5 | Same assertion message pattern (different trigger) |

### Relationship

This bug is a **specific manifestation** of the broader string type support gap:
- **#8283** reports the general problem: MooreToCore cannot handle string types
- **Current bug** shows a specific crash path: string type used as module output port
- **#8930** shows the same assertion pattern with a different unsupported type (`real`)

## Additional Information

### Test Case Reduction

- **Original**: 274 bytes
- **Minimal**: 45 bytes
- **Reduction**: **83.6%**

### Keywords

`string` `port` `MooreToCore` `getModulePortInfo` `dyn_cast` `InOutType` `DynamicStringType` `assertion` `type conversion` `SVModuleOp`

### Source Files

- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Main conversion pass
- `include/circt/Dialect/Moore/MooreTypes.td` - Moore type definitions
- `include/circt/Dialect/HW/HWTypes.h` - HW type definitions
- `include/circt/Dialect/Sim/SimTypes.td` - Sim dialect types
