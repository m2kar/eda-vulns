# [MooreToCore] Crash with `string` type as module port

## Summary

`circt-verilog` crashes with assertion failure when processing SystemVerilog modules with `string` type input ports. The crash occurs in the `MooreToCore` conversion pass when `getModulePortInfo()` attempts to convert port types.

## Test Case

### Minimal Reproducer

```systemverilog
module test_module(input string a);
endmodule
```

### Command to Reproduce

```bash
circt-verilog --ir-hw test_module.sv
```

### Crash Output

```
PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
Stack dump:
0.	Program arguments: circt-verilog --ir-hw test_module.sv
...
#16 0x00007fd0c0e26231 (anonymous namespace)::MooreToCorePass::runOnOperation() MooreToCore.cpp:0:0
...
#4 0x00007fd0c0e548ae (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, circt::moore::SVModuleOpAdaptor, mlir::ConversionPatternRewriter&) const MooreToCore.cpp:0:0
...
circt-verilog: .../llvm/include/llvm/Support/Casting.h:650: decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]: Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
Aborted (core dumped)
```

## Root Cause

The crash occurs in `MooreToCore.cpp`:

1. **Parser**: Creates `moore::StringType` for the `input string a` port
2. **getModulePortInfo()**: Calls `typeConverter.convertType(port.type)` for each port
3. **Type Conversion**: Returns `nullptr` for `StringType` (no conversion registered in CIRCT 1.139.0)
4. **ModulePortInfo Constructor**: Calls `sanitizeInOut()` which performs `dyn_cast<hw::InOutType>` on null type
5. **CRASH**: Assertion failure in `dyn_cast` because type is null

### Code Location

- **File**: `lib/Conversion/MooreToCore/MooreToCore.cpp:259`
- **Function**: `getModulePortInfo(const TypeConverter &typeConverter, SVModuleOp op)`
- **Crash Site**: `hw::ModulePortInfo(ports)` → `sanitizeInOut()` → `dyn_cast<hw::InOutType>`

### Missing Type Conversion

In CIRCT 1.139.0, the type converter does not have a registered conversion for `moore::StringType`. The current master branch appears to have added this conversion:

```cpp
typeConverter.addConversion([&](StringType type) {
  return sim::DynamicStringType::get(type.getContext());
});
```

## Analysis

### Syntax Validity

The test case is **valid SystemVerilog** per IEEE 1800-2017. String types are allowed in module ports for simulation purposes.

### Cross-Tool Validation

| Tool | Status | Notes |
|-------|--------|-------|
| **Slang** | ✅ Pass | "Build succeeded: 0 errors, 0 warnings" |
| **Verilator** | ✅ Pass | "lint-only succeeded with no errors" |
| **Icarus** | ❌ Fail | Does not support string ports (synthesis-oriented limitation) |

### Related Issues

**Highly Related**: #8283 - `[ImportVerilog] Cannot compile forward decleared string type`

- **Similarity Score**: 8.5/10
- **Root Cause**: Same - `moore::StringType` has no registered type conversion
- **Difference**: #8283 reports string as a local variable, this issue reports string as a module port
- **Recommendation**: Consider merging or adding this test case to #8283

## Suggested Fixes

### Priority 1: Add Type Conversion
Register the missing conversion in `populateTypeConversion()`:

```cpp
typeConverter.addConversion([&](StringType type) {
  return sim::DynamicStringType::get(type.getContext());
});
```

### Priority 2: Add Null Check
Validate type conversion result in `getModulePortInfo()`:

```cpp
for (auto port : moduleTy.getPorts()) {
  Type portTy = typeConverter.convertType(port.type);
  if (!portTy) {
    op.emitError() << "unsupported port type: " << port.type;
    return failure();
  }
  // ... rest of code
}
```

### Priority 3: Defensive Fix
Add null check before `dyn_cast` in `sanitizeInOut()`:

```cpp
if (p.type && isa<hw::InOutType>(p.type)) {
  // ...
}
```

## Version Information

- **CIRCT Version**: 1.139.0
- **Crash ID**: b917a28cc834
- **Likely Fixed In**: master branch (type conversion for StringType exists)

## Additional Notes

- The string type is a dynamic, variable-length data type in SystemVerilog
- It is not synthesizable for hardware, but is valid for simulation
- CIRCT should handle this gracefully - either by supporting the conversion or by emitting a proper error diagnostic
- Crashing with assertion failure is a bug that prevents proper error reporting

## Attachments

- [test_module.sv](attachment://test_module.sv) - Minimal test case (2 lines)
- [error.log](attachment://error.log) - Full crash backtrace
- [analysis.json](attachment://analysis.json) - Detailed root cause analysis
