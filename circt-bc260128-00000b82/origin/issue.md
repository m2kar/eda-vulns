# Assertion failure in `dyn_cast` when lowering unpacked array of strings in Moore-to-Core conversion

## Issue Type
**Bug**

## Summary
CIRCT crashes with an assertion failure when attempting to lower a Moore SystemVerilog module containing a port declared as an **unpacked array of strings** (`output string s[1:0]`). The crash occurs in `circt::hw::ModulePortInfo::sanitizeInOut()` during Moore-to-Core conversion, triggered by an unsafe `dyn_cast` on a null/non-existent `mlir::Type`.

## Description

### Problem Statement
When processing a module port declared as `output string s[1:0]` (an unpacked array of SystemVerilog strings), the Moore-to-Core dialect conversion pass yields a null `mlir::Type` for this unsupported port type. The subsequent call to `ModulePortInfo::sanitizeInOut()` performs an unchecked `dyn_cast<circt::hw::InOutType>` on this null type, triggering an assertion failure:

```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed
```

### Root Cause Analysis
The root cause lies in the interaction between:

1. **Unsupported Type Conversion**: The Moore-to-Core type converter returns a null/empty `mlir::Type` for unpacked arrays of strings (which are not representable as hardware ports).

2. **Missing Validation**: The `getModulePortInfo()` function in `MooreToCore.cpp` constructs `PortInfo` entries without validating that the type conversion succeeded. It blindly builds the port list regardless of null types.

3. **Unsafe Type Cast**: The `ModulePortInfo::sanitizeInOut()` method performs an unchecked `dyn_cast` on this null type without first confirming the type's existence, causing the assertion failure.

### Stack Trace Analysis
The crash originates from the following call chain:

- **#22**: `SVModuleOpConversion::matchAndRewrite()` (MooreToCore.cpp:276)
- **#21**: `getModulePortInfo()` (MooreToCore.cpp:259) — builds port list with null type
- **#17**: `ModulePortInfo::sanitizeInOut()` (PortImplementation.h:177) — performs unsafe dyn_cast ← **Assertion failure here**

The full stack trace shows the Moore-to-Core dialect conversion framework (ConversionPattern, OperationLegalizer, etc.) triggering the pattern match.

## Minimal Reproducer

### Test Case
```systemverilog
module bug(
  output string s[1:0]
);
endmodule
```

### Reproduction Command
```bash
/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw bug.sv
```

### Expected Behavior
CIRCT should either:
1. Successfully lower the unpacked array of strings to a supported hardware representation, OR
2. Emit a clear diagnostic error message indicating that unpacked arrays of strings are not supported as module ports.

### Actual Behavior
CIRCT crashes with an assertion failure:
```
circt-verilog: /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/Casting.h:650: 
decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]: 
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

Exit code: 134 (SIGABRT)

## Backtrace

```
 #0 0x000055d771ab232f llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/lib/Support/Unix/Signals.inc:842:13
 #1 0x000055d771ab32e9 llvm::sys::RunSignalHandlers() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/lib/Support/Signals.cpp:109:18
 #2 0x000055d771ab32e9 SignalHandler(int, siginfo_t*, void*) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/lib/Support/Unix/Signals.inc:412:3
 ...
#17 0x000055d77022d874 llvm::DefaultDoCastIfPossible<circt::hw::InOutType, mlir::Type, llvm::CastInfo<circt::hw::InOutType, mlir::Type, void>>::doCastIfPossible(mlir::Type) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/Casting.h:311:10
#18 0x000055d77022d874 decltype(auto) llvm::dyn_cast<circt::hw::InOutType, mlir::Type>(mlir::Type&) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/Casting.h:651:10
#19 0x000055d77022d874 circt::hw::ModulePortInfo::sanitizeInOut() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/include/circt/Dialect/HW/PortImplementation.h:177:24
#20 0x000055d770525753 llvm::SmallVectorTemplateCommon<circt::hw::PortInfo, void>::begin() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/ADT/SmallVector.h:271:45
#21 0x000055d770525753 (anonymous namespace)::getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Conversion/MooreToCore/MooreToCore.cpp:259:1
#22 0x000055d770525753 (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, circt::moore::SVModuleOpAdaptor, mlir::ConversionPatternRewriter&) const /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Conversion/MooreToCore/MooreToCore.cpp:276:32
#23 0x000055d770525a81 llvm::LogicalResult mlir::ConversionPattern::dispatchTo1To1<mlir::OpConversionPattern<circt::moore::SVModuleOp>, circt::moore::SVModuleOp>(mlir::OpConversionPattern<circt::moore::SVModuleOp> const&, circt::moore::SVModuleOp, circt::moore::SVModuleOp::GenericAdaptor<llvm::ArrayRef<mlir::ValueRange>>, mlir::ConversionPatternRewriter&) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/Transforms/DialectConversion.h:0:15
#24 0x000055d770525515 mlir::OpConversionPattern<circt::moore::SVModuleOp>::matchAndRewrite(mlir::Operation*, llvm::ArrayRef<mlir::ValueRange>, mlir::ConversionPatternRewriter&) const /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/Transforms/DialectConversion.h:715:3
...
#42 0x000055d7704ef832 (anonymous namespace)::MooreToCorePass::runOnOperation() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Conversion/MooreToCore/MooreToCore.cpp:2571:14
```

## Environment
- **CIRCT Version**: 1.139.0
- **Tool**: circt-verilog
- **Dialects Involved**: Moore, HW, Core
- **Relevant Pass**: MooreToCore dialect conversion

## Related Issues
This crash is a manifestation of the underlying feature gap tracked in **#8276** — `[MooreToCore] Support for UnpackedArrayType emission`. 

The crash occurs because unpacked arrays (including unpacked arrays of strings) are not fully supported in Moore-to-Core conversion. When the type converter fails to produce a valid `mlir::Type`, the port creation code does not handle the failure gracefully, leading to this assertion.

**Note**: Issue #8276 is an OPEN feature request that blocks the proper support for unpacked array types in the Moore dialect. This crash is directly caused by the missing implementation described there.

## Additional Context

### Validation Results
- ✅ **Crash Reproduced**: Yes, exact signature match
- ✅ **Syntax Valid**: Verilator accepts the test case without diagnostics
- ✅ **Minimal**: Only 5 lines needed to trigger the crash

### Classification
- **Type**: Assertion failure during dialect conversion
- **Severity**: High (causes tool crash)
- **Reproducibility**: Deterministic, 100%

### Recommended Fix Strategy
To resolve this issue, CIRCT developers should:

1. **Add type validation in `getModulePortInfo()`**: Check if the type converter returns a null/empty type. If so, emit a diagnostic with the port name and type information, then return failure.

2. **Add defensive checks in `sanitizeInOut()`**: Guard `dyn_cast` operations with explicit null checks. If the type is null, emit a clearer diagnostic and handle gracefully.

3. **Extend type support** (longer-term): Implement proper lowering for unpacked array types in Moore-to-Core (addresses #8276), or explicitly document and reject these types with user-friendly error messages.

### References
- **Stack Trace Assertion**: `llvm/Support/Casting.h:650`
- **Port Info Implementation**: `include/circt/Dialect/HW/PortImplementation.h:177`
- **Moore-to-Core Conversion**: `lib/Conversion/MooreToCore/MooreToCore.cpp:259, 276`
- **Blocking Feature Issue**: #8276
