# [MooreToCore] Assertion failure when module has string type output port

## Bug Description

`circt-verilog` crashes with an assertion failure when processing a SystemVerilog module that has a `string` type output port. The crash occurs during the MooreToCore conversion pass when the `getModulePortInfo()` function fails to properly handle cases where type conversion returns an invalid/empty type, causing a `dyn_cast` assertion failure in `ModulePortInfo::sanitizeInOut()`.

This is a valid SystemVerilog construct per IEEE 1800-2017 Section 6.16 (String data type). Both **slang** and **verilator** accept the code without errors.

## Steps to Reproduce

1. Save the following test case as `test.sv`
2. Run: `circt-verilog --ir-hw test.sv`

## Test Case

```systemverilog
module test_module(output string a);
endmodule
```

## Expected Behavior

CIRCT should either:
1. Support simulation constructs like string ports appropriately, OR
2. Emit a proper diagnostic error message indicating that string ports are not supported for hardware synthesis

## Actual Behavior

The tool crashes with an assertion failure:

```
PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
Stack dump:
0.	Program arguments: circt-verilog --ir-hw bug.sv
 #0 0x00007fe810a738a8 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/opt/firtool-1.139.0/bin/../lib/libLLVMSupport.so+0x2008a8)
 #1 0x00007fe810a712f5 llvm::sys::RunSignalHandlers() (/opt/firtool-1.139.0/bin/../lib/libLLVMSupport.so+0x1fe2f5)
 #2 0x00007fe810a74631 SignalHandler(int, siginfo_t*, void*) Signals.cpp:0:0
 #3 0x00007fe810581330 (/lib/x86_64-linux-gnu/libc.so.6+0x45330)
 #4 0x00007fe814c958ae (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, circt::moore::SVModuleOpAdaptor, mlir::ConversionPatternRewriter&) const MooreToCore.cpp:0:0
 #5 0x00007fe814c95b93 llvm::LogicalResult mlir::ConversionPattern::dispatchTo1To1<...> (/opt/firtool-1.139.0/bin/../lib/libCIRCTMooreToCore.so+0x50b93)
 ...
#16 0x00007fe814c67231 (anonymous namespace)::MooreToCorePass::runOnOperation() MooreToCore.cpp:0:0
```

## Root Cause Analysis

### Crash Location
- **File**: `lib/Conversion/MooreToCore/MooreToCore.cpp`
- **Function**: `getModulePortInfo()` → calls `ModulePortInfo::sanitizeInOut()`
- **Assertion**: `detail::isPresent(Val) && "dyn_cast on a non-existent value"`

### Analysis

The root cause is **missing validation of type conversion result** in `getModulePortInfo()`:

```cpp
// MooreToCore.cpp:233-259
static hw::ModulePortInfo getModulePortInfo(const TypeConverter &typeConverter,
                                            SVModuleOp op) {
  // ...
  for (auto port : moduleTy.getPorts()) {
    Type portTy = typeConverter.convertType(port.type);  // <-- May return empty Type!
    // ...
    ports.push_back(hw::PortInfo({{port.name, portTy, port.dir}, ...}));  // <-- Stores empty Type
  }
  return hw::ModulePortInfo(ports);  // <-- Constructor calls sanitizeInOut()
}
```

When `typeConverter.convertType()` fails for the `string` port type (returns empty `Type`), this invalid type is passed to `sanitizeInOut()` which calls `dyn_cast<hw::InOutType>(p.type)` on the empty type, triggering the assertion.

### Suggested Fix

Add validation in `getModulePortInfo()` to check if `typeConverter.convertType()` returns a valid type:

```cpp
Type portTy = typeConverter.convertType(port.type);
if (!portTy) {
  // Emit proper diagnostic error
  return failure();
}
```

## Environment

- **CIRCT Version**: firtool-1.139.0 (LLVM 22.0.0git)
- **OS**: Linux 5.15.0 (x86_64)

## Cross-Tool Validation

| Tool | Version | Status | Notes |
|------|---------|--------|-------|
| slang | 10.0.6+3d7e6cd2e | ✅ Pass | Build succeeded: 0 errors, 0 warnings |
| verilator | 5.022 | ✅ Pass | No errors |
| iverilog | - | ❌ Error | "Port with type string is not supported" (tool limitation, not syntax error) |

This confirms the test case is **syntactically valid** SystemVerilog per IEEE 1800-2017.

## Related Issues

- #8930 - [MooreToCore] Crash with sqrt/floor
  - **Same assertion message**: `dyn_cast on a non-existent value`
  - **Different crash location**: `ConversionOpConversion::matchAndRewrite` (real type conversion)
  - **Relationship**: Both expose the same underlying problem - MooreToCore's type converter returns empty/invalid types that are passed to `dyn_cast` without validation

## Additional Context

<details>
<summary>Full Stack Trace</summary>

```
PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
Stack dump:
0.	Program arguments: circt-verilog --ir-hw bug.sv
 #0 0x00007fe810a738a8 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/opt/firtool-1.139.0/bin/../lib/libLLVMSupport.so+0x2008a8)
 #1 0x00007fe810a712f5 llvm::sys::RunSignalHandlers() (/opt/firtool-1.139.0/bin/../lib/libLLVMSupport.so+0x1fe2f5)
 #2 0x00007fe810a74631 SignalHandler(int, siginfo_t*, void*) Signals.cpp:0:0
 #3 0x00007fe810581330 (/lib/x86_64-linux-gnu/libc.so.6+0x45330)
 #4 0x00007fe814c958ae (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, circt::moore::SVModuleOpAdaptor, mlir::ConversionPatternRewriter&) const MooreToCore.cpp:0:0
 #5 0x00007fe814c95b93 llvm::LogicalResult mlir::ConversionPattern::dispatchTo1To1<mlir::OpConversionPattern<circt::moore::SVModuleOp>, circt::moore::SVModuleOp>(mlir::OpConversionPattern<circt::moore::SVModuleOp> const&, circt::moore::SVModuleOp, circt::moore::SVModuleOp::GenericAdaptor<llvm::ArrayRef<mlir::ValueRange>>, mlir::ConversionPatternRewriter&) (/opt/firtool-1.139.0/bin/../lib/libCIRCTMooreToCore.so+0x50b93)
 #6 0x00007fe814c95530 mlir::OpConversionPattern<circt::moore::SVModuleOp>::matchAndRewrite(mlir::Operation*, llvm::ArrayRef<mlir::ValueRange>, mlir::ConversionPatternRewriter&) const (/opt/firtool-1.139.0/bin/../lib/libCIRCTMooreToCore.so+0x50530)
 #7 0x00007fe812fa7438 mlir::ConversionPattern::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const (/opt/firtool-1.139.0/bin/../lib/libMLIRTransformUtils.so+0x2a438)
 #8 0x00007fe812f718ed void llvm::function_ref<void ()>::callback_fn<mlir::PatternApplicator::matchAndRewrite(...)> PatternApplicator.cpp:0:0
 #9 0x00007fe812f6e774 mlir::PatternApplicator::matchAndRewrite(...) (/opt/firtool-1.139.0/bin/../lib/libMLIRRewrite.so+0x7774)
#10 0x00007fe812fa8c6f (anonymous namespace)::OperationLegalizer::legalize(mlir::Operation*) DialectConversion.cpp:0:0
#11 0x00007fe812fa8470 mlir::OperationConverter::convert(mlir::Operation*, bool) (/opt/firtool-1.139.0/bin/../lib/libMLIRTransformUtils.so+0x2b470)
#12 0x00007fe812fa8dae mlir::OperationConverter::convertOperations(llvm::ArrayRef<mlir::Operation*>) (/opt/firtool-1.139.0/bin/../lib/libMLIRTransformUtils.so+0x2bdae)
#13 0x00007fe812fb68e4 void llvm::function_ref<void ()>::callback_fn<applyConversion(...)> DialectConversion.cpp:0:0
#14 0x00007fe812fadf7d applyConversion(...) DialectConversion.cpp:0:0
#15 0x00007fe812fae0fe mlir::applyFullConversion(mlir::Operation*, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig) (/opt/firtool-1.139.0/bin/../lib/libMLIRTransformUtils.so+0x310fe)
#16 0x00007fe814c67231 (anonymous namespace)::MooreToCorePass::runOnOperation() MooreToCore.cpp:0:0
#17 0x00007fe812d632a5 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) (/opt/firtool-1.139.0/bin/../lib/libMLIRPass.so+0x172a5)
#18 0x00007fe812d667a9 mlir::PassManager::run(mlir::Operation*) (/opt/firtool-1.139.0/bin/../lib/libMLIRPass.so+0x1a7a9)
#19 0x0000561bab9605d0 executeWithSources(mlir::MLIRContext*, llvm::SourceMgr&) circt-verilog.cpp:0:0
#20 0x0000561bab95bdd5 execute(mlir::MLIRContext*) circt-verilog.cpp:0:0
#21 0x0000561bab95b4b8 main (/opt/firtool-1.139.0/bin/circt-verilog+0x84b8)
#22 0x00007fe8105661ca __libc_start_call_main ./csu/../sysdeps/nptl/libc_start_call_main.h:74:3
#23 0x00007fe81056628b call_init ./csu/../csu/libc-start.c:128:20
#24 0x00007fe81056628b __libc_start_main ./csu/../csu/libc-start.c:347:5
#25 0x0000561bab95ab05 _start (/opt/firtool-1.139.0/bin/circt-verilog+0x7b05)
```

</details>

---
*This issue was generated with assistance from an automated bug reporter.*
