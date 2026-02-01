# [Moore] Assertion failure when module has string type output port

## Bug Description

`circt-verilog` crashes with an assertion failure when processing a SystemVerilog module that has a `string` type output port. The crash occurs during the MooreToCore conversion pass when the `getModulePortInfo()` function fails to properly handle cases where type conversion returns an invalid/empty type, causing a `dyn_cast` assertion failure in `ModulePortInfo::sanitizeInOut()`.

This is a valid SystemVerilog construct per IEEE 1800-2017 Section 6.16 (String data type). The compiler should either:
1. Successfully lower the string type port to a simulation-compatible representation, or
2. Emit a clear diagnostic error message indicating the limitation

Instead, it crashes with an internal assertion failure.

## Steps to Reproduce

**Test Case (minimized)**:
```systemverilog
module m(output string s);
endmodule
```

**Command**:
```bash
circt-verilog --ir-hw bug.sv
```

**Expected Behavior**: Either successful compilation or a diagnostic error about unsynthesizable string ports

**Actual Behavior**: Assertion failure and crash

## Error Output

```
circt-verilog: /path/to/llvm/include/llvm/Support/Casting.h:650: decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]: Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
Stack dump:
...
#17 0x... circt::hw::ModulePortInfo::sanitizeInOut() .../PortImplementation.h:177:24
#21 0x... (anonymous namespace)::getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp) .../MooreToCore.cpp:259:1
#22 0x... (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, circt::moore::SVModuleOpAdaptor, mlir::ConversionPatternRewriter&) const .../MooreToCore.cpp:276:32
...
Aborted (core dumped)
```

## Root Cause Analysis

### Crash Mechanism

1. **Type Conversion Issue**: The MooreToCore type converter (lines 2277-2278) converts `moore::StringType` ports to `sim::DynamicStringType`:
   ```cpp
   typeConverter.addConversion([&](StringType type) {
     return sim::DynamicStringType::get(type.getContext());
   });
   ```

2. **Invalid HW Dialect Type**: `sim::DynamicStringType` is not a valid type for hardware module ports in the HW dialect.

3. **Assertion Failure**: When `hw::ModulePortInfo` is constructed with this port type, `sanitizeInOut()` method attempts to `dyn_cast<hw::InOutType>` on ALL port types without validity checks. The assertion `detail::isPresent(Val)` fails because the `sim::DynamicStringType` cannot be cast as `hw::InOutType`.

### Affected Components

| Component | File | Issue |
|-----------|------|-------|
| TypeConverter | MooreToCore.cpp:2277 | Converts StringType to sim::DynamicStringType |
| getModulePortInfo | MooreToCore.cpp:243 | No null/validity check on converted port type |
| sanitizeInOut | PortImplementation.h:177 | Assumes all port types can be safely cast |

## Proposed Fix

### Option 1: Validate Port Types in getModulePortInfo()
```cpp
for (auto port : moduleTy.getPorts()) {
  Type portTy = typeConverter.convertType(port.type);
  if (!portTy || !hw::isHWValueType(portTy)) {
    // Emit error: unsupported port type
    op->emitError() << "unsupported port type: " << port.type;
    return failure();
  }
  // ... proceed with valid type
}
```

### Option 2: Guard in sanitizeInOut()
```cpp
void sanitizeInOut() {
  for (auto &p : ports)
    if (p.type && isa<hw::InOutType>(p.type)) {
      auto inout = cast<hw::InOutType>(p.type);
      p.type = inout.getElementType();
      p.dir = ModulePort::Direction::InOut;
    }
}
```

### Option 3: Reject StringType Ports Early
Add validation in MooreToCorePass to reject modules with string-type ports with clear diagnostics.

## Additional Information

### Syntax Validation
| Tool | Status | Notes |
|------|--------|-------|
| slang | ✅ PASS | Valid SystemVerilog syntax |
| verilator | ✅ PASS | Valid SystemVerilog syntax |

### Impact
- **Severity**: High (crash/assertion failure)
- **Reproducibility**: 100% with string-type module ports
- **Affected Versions**: CIRCT 1.139.0 (and likely earlier versions)
- **Testcase ID**: bc260129-00001473
- **Reduction**: 92.0% (25 → 2 lines)

### Duplicate Issue Check

**Note**: This analysis found a highly similar existing issue:
- **Issue #9572**: "[Moore] Assertion failure when module has string type output port"
- **Similarity Score**: 15.0/10.0 (threshold exceeded)
- **Status**: OPEN

This minimized testcase may serve as additional evidence for the existing issue.

### Environment
- CIRCT Version: 1.139.0
- LLVM Version: 22.0.0git
- System: Linux x86_64
