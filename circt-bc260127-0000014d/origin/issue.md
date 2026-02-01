# [MooreToCore] Crash on module with string type port

## Summary

CIRCT crashes with an assertion failure when compiling a SystemVerilog module with a `string` type input port. The MooreToCore dialect conversion fails to handle the string type, resulting in a null type being passed to a `dyn_cast` operation in `ModulePortInfo::sanitizeInOut()`.

## Minimal Test Case

```systemverilog
module test(input string a);
endmodule
```

## Reproduction

**Command:**
```bash
circt-verilog --ir-hw bug.sv
```

**Environment:**
- CIRCT Version: 1.139.0
- Toolchain: firtool-1.139.0

## Expected vs. Actual Behavior

**Expected**: Either successfully compile the module or emit a clear error message explaining that string type ports are not supported.

**Actual**: Assertion failure with stack trace starting at `SVModuleOpConversion::matchAndRewrite()`.

## Error Details

### Assertion Failure
```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

### Stack Trace

The crash occurs during the MooreToCore pass conversion:

```
Stack dump:
 #0 0x00007f0204f318a8 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int)
 #4 0x00007f02091538ae (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, circt::moore::SVModuleOpAdaptor, mlir::ConversionPatternRewriter&) const MooreToCore.cpp:0:0
...
#16 0x00007f0209125231 (anonymous namespace)::MooreToCorePass::runOnOperation() MooreToCore.cpp:0:0
```

The crash is triggered when attempting to dyn_cast a null type in `ModulePortInfo::sanitizeInOut()`:

```cpp
void sanitizeInOut() {
  for (auto &p : ports)
    if (auto inout = dyn_cast<hw::InOutType>(p.type)) {  // CRASH HERE
      p.type = inout.getElementType();
      p.dir = ModulePort::Direction::InOut;
    }
}
```

## Root Cause

The MooreToCore type converter fails to convert the SystemVerilog `string` type to a valid HW type. The conversion returns a null/invalid `mlir::Type`, which is then passed to `dyn_cast<hw::InOutType>()` in the `sanitizeInOut()` function.

### Key Issues

1. **Type Conversion Failure**: The `TypeConverter` in MooreToCore does not have a conversion rule for the `string` type, which is a dynamic, variable-length type without direct hardware representation.

2. **Silent Failure**: The type conversion failure is not properly handled. The code continues with a null type instead of emitting a diagnostic error.

3. **Unprotected Casting**: The `sanitizeInOut()` function assumes all port types are valid and directly casts them without null checks. The LLVM `dyn_cast` macro includes an assertion that fires when the value is not present.

### Code Path

```
SVModuleOpConversion::matchAndRewrite()
  └─> getModulePortInfo(TypeConverter, SVModuleOp)
        └─> For each port, convert type using TypeConverter
        └─> Type conversion for 'string' → null/invalid
        └─> Create PortInfo with null type
        └─> Construct ModulePortInfo(ports)
              └─> sanitizeInOut()
                    └─> dyn_cast<InOutType>(p.type) ← ASSERTION FAILURE
```

## Validation

This is a valid bug report confirmed through cross-tool validation:

| Tool | Version | Result | Notes |
|------|---------|--------|-------|
| **Verilator** | 5.022 | ✅ Accepts | No errors or warnings |
| **Slang** | 10.0.6 | ✅ Accepts | Build succeeded: 0 errors, 0 warnings |

Both Verilator and Slang accept this as valid SystemVerilog code, confirming that CIRCT is incorrectly rejecting valid input.

## Related Issues

**Issue #8930**: [MooreToCore] Crash with sqrt/floor
- **Similarity**: LIKELY DUPLICATE (same root cause)
- **Description**: Same assertion failure (`dyn_cast on a non-existent value`) in MooreToCore type conversion
- **Difference**: Triggered by real/sqrt/floor operations instead of string ports
- **Impact**: This report provides a simpler, more focused test case that isolates the port-specific conversion failure

**Issue #8283**: [ImportVerilog] Cannot compile forward declared string type
- **Related**: SystemVerilog string type handling

**Issue #8332**: [MooreToCore] Support for StringType from moore to llvm dialect
- **Related**: String type conversion feature request

## Suggested Fixes

1. **Option 1**: Emit a proper diagnostic error when type conversion fails in `getModulePortInfo()` instead of continuing with a null type.

2. **Option 2**: Add a null check in `sanitizeInOut()` before calling `dyn_cast`:
   ```cpp
   void sanitizeInOut() {
     for (auto &p : ports) {
       if (!p.type) {
         emitError("Port has unconvertible type");
         continue;
       }
       if (auto inout = dyn_cast<hw::InOutType>(p.type)) {
         p.type = inout.getElementType();
         p.dir = ModulePort::Direction::InOut;
       }
     }
   }
   ```

3. **Option 3**: Implement proper type conversion for `string` type ports (design decision).

4. **Option 4**: Reject unsupported types like `string` earlier in the ImportVerilog phase with a clear error message.

## Additional Information

### CIRCT Version
- Version: 1.139.0
- Build: firtool-1.139.0

### Files Affected
- `lib/Conversion/MooreToCore/MooreToCore.cpp` (line 259, function `getModulePortInfo`)
- `include/circt/Dialect/HW/PortImplementation.h` (line 177, function `ModulePortInfo::sanitizeInOut`)

### Compilation Environment
- Platform: Linux x86_64
- LLVM Support Library: libLLVMSupport.so (version 22.0.0git)
- MLIR Transform Utils: libMLIRTransformUtils.so

### Testcase Analysis
The minimal testcase reduction achieved **80.9% size reduction** (204 → 39 bytes) while maintaining crash reproducibility. This indicates the core issue is purely the `string` type port declaration, with no dependence on:
- Module logic or assignments
- Other port types
- Body complexity

This simplicity makes it an ideal test case for regression testing.
