<!-- Title: [Moore] Assertion failure when converting module with string type port in MooreToCore -->

## Description

`circt-verilog` crashes with an assertion failure when processing a SystemVerilog module that has a `string` type output port. The crash occurs in the `MooreToCore` conversion pass when attempting to create HW module port information.

**Root Cause**: MooreToCore conversion converts Moore `StringType` to `sim::DynamicStringType`, but HW dialect does not accept this type for module ports (`isHWValueType()` returns false). The conversion pass lacks validation to catch this unsupported case, leading to an assertion failure instead of a proper diagnostic error.

**Crash Type**: Assertion failure  
**Dialect**: Moore  
**Failing Pass**: MooreToCorePass (`SVModuleOpConversion` → `getModulePortInfo`)

## Steps to Reproduce

1. Save the test case below as `test.sv`
2. Run:
   ```bash
   circt-verilog --ir-hw test.sv
   ```

## Minimal Test Case

```systemverilog
module test(output string str_out);
endmodule
```

## Error Output

```
PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
Stack dump:
0.	Program arguments: /opt/firtool-1.139.0/bin/circt-verilog --ir-hw bug.sv
 #0 0x00007fb020d298a8 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/opt/firtool-1.139.0/bin/../lib/libLLVMSupport.so+0x2008a8)
 #1 0x00007fb020d272f5 llvm::sys::RunSignalHandlers() (/opt/firtool-1.139.0/bin/../lib/libLLVMSupport.so+0x1fe2f5)
 #2 0x00007fb020d2a631 SignalHandler(int, siginfo_t*, void*) Signals.cpp:0:0
 #3 0x00007fb020837330 (/lib/x86_64-linux-gnu/libc.so.6+0x45330)
 #4 0x00007fb024f4b8ae (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(...) MooreToCore.cpp:0:0
 ...
```

## Root Cause Analysis

### Assertion Message
```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

### Key Stack Frames
```
#4  SVModuleOpConversion::matchAndRewrite(...) MooreToCore.cpp:0:0
#13 getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp) MooreToCore.cpp:259
#35 MooreToCorePass::runOnOperation() MooreToCore.cpp:2571
```

### Analysis

1. **Type Conversion Chain**:
   - Moore `StringType` → `typeConverter.convertType()` → `sim::DynamicStringType`

2. **Crash Mechanism**:
   - `getModulePortInfo()` calls `convertType()` for each port type
   - Returns `sim::DynamicStringType` for string port
   - `HWModuleOp::build()` attempts to use this type
   - `isHWValueType(DynamicStringType)` returns false
   - Assertion fails when trying to create `InOutType` with unsupported inner type

3. **Missing Validation**:
   - `getModulePortInfo()` unconditionally uses `convertType()` result without checking if the result is a valid HW port type

### Suggested Fix Directions

1. Add type validation in `getModulePortInfo()`:
   ```cpp
   Type portTy = typeConverter.convertType(port.type);
   if (!portTy || !hw::isHWValueType(portTy)) {
     // Emit error: unsupported port type
     return {};
   }
   ```

2. Or emit a proper diagnostic before the conversion pass for unsupported port types.

## Validation

| Tool | Status | Notes |
|------|--------|-------|
| Slang | ✅ pass | Build succeeded: 0 errors, 0 warnings |
| Verilator | ✅ pass | Lint passed |
| Icarus | ❌ error | Tool limitation: "string type ports not supported" |

The test case uses valid IEEE 1800-2017 SystemVerilog syntax. The `string` type as a module port is explicitly supported by the standard (Section 6.16, Section 23.2.2).

**Expected Behavior**: CIRCT should either:
1. Support string type ports (convert to appropriate simulation type), OR
2. Reject with a clear error message like "string type ports are not supported for hardware synthesis"

The current behavior (assertion failure/crash) is incorrect.

## Environment

- **CIRCT Version**: firtool-1.139.0 (LLVM 22.0.0git)
- **OS**: Linux
- **Architecture**: x86_64

## Stack Trace

<details>
<summary>Click to expand full stack trace</summary>

```
PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
Stack dump:
0.	Program arguments: /opt/firtool-1.139.0/bin/circt-verilog --ir-hw bug.sv
 #0 0x00007fb020d298a8 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/opt/firtool-1.139.0/bin/../lib/libLLVMSupport.so+0x2008a8)
 #1 0x00007fb020d272f5 llvm::sys::RunSignalHandlers() (/opt/firtool-1.139.0/bin/../lib/libLLVMSupport.so+0x1fe2f5)
 #2 0x00007fb020d2a631 SignalHandler(int, siginfo_t*, void*) Signals.cpp:0:0
 #3 0x00007fb020837330 (/lib/x86_64-linux-gnu/libc.so.6+0x45330)
 #4 0x00007fb024f4b8ae (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, circt::moore::SVModuleOpAdaptor, mlir::ConversionPatternRewriter&) const MooreToCore.cpp:0:0
 #5 0x00007fb024f4bb93 llvm::LogicalResult mlir::ConversionPattern::dispatchTo1To1<mlir::OpConversionPattern<circt::moore::SVModuleOp>, circt::moore::SVModuleOp>(mlir::OpConversionPattern<circt::moore::SVModuleOp> const&, circt::moore::SVModuleOp, circt::moore::SVModuleOp::GenericAdaptor<llvm::ArrayRef<mlir::ValueRange>>, mlir::ConversionPatternRewriter&) (/opt/firtool-1.139.0/bin/../lib/libCIRCTMooreToCore.so+0x50b93)
 #6 0x00007fb024f4b530 mlir::OpConversionPattern<circt::moore::SVModuleOp>::matchAndRewrite(mlir::Operation*, llvm::ArrayRef<mlir::ValueRange>, mlir::ConversionPatternRewriter&) const (/opt/firtool-1.139.0/bin/../lib/libCIRCTMooreToCore.so+0x50530)
 #7 0x00007fb02325d438 mlir::ConversionPattern::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const (/opt/firtool-1.139.0/bin/../lib/libMLIRTransformUtils.so+0x2a438)
 #8 0x00007fb0232278ed void llvm::function_ref<void ()>::callback_fn<mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>)::$_0>(long) PatternApplicator.cpp:0:0
 #9 0x00007fb023224774 mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>) (/opt/firtool-1.139.0/bin/../lib/libMLIRRewrite.so+0x7774)
#10 0x00007fb02325ec6f (anonymous namespace)::OperationLegalizer::legalize(mlir::Operation*) DialectConversion.cpp:0:0
#11 0x00007fb02325e470 mlir::OperationConverter::convert(mlir::Operation*, bool) (/opt/firtool-1.139.0/bin/../lib/libMLIRTransformUtils.so+0x2b470)
#12 0x00007fb02325edae mlir::OperationConverter::convertOperations(llvm::ArrayRef<mlir::Operation*>) (/opt/firtool-1.139.0/bin/../lib/libMLIRTransformUtils.so+0x2bdae)
#13 0x00007fb02326c8e4 void llvm::function_ref<void ()>::callback_fn<applyConversion(llvm::ArrayRef<mlir::Operation*>, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig, (anonymous namespace)::OpConversionMode)::$_0>(long) DialectConversion.cpp:0:0
#14 0x00007fb023263f7d applyConversion(llvm::ArrayRef<mlir::Operation*>, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig, (anonymous namespace)::OpConversionMode) DialectConversion.cpp:0:0
#15 0x00007fb0232640fe mlir::applyFullConversion(mlir::Operation*, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig) (/opt/firtool-1.139.0/bin/../lib/libMLIRTransformUtils.so+0x310fe)
#16 0x00007fb024f1d231 (anonymous namespace)::MooreToCorePass::runOnOperation() MooreToCore.cpp:0:0
#17 0x00007fb0230192a5 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) (/opt/firtool-1.139.0/bin/../lib/libMLIRPass.so+0x172a5)
#18 0x00007fb02301c7a9 mlir::PassManager::run(mlir::Operation*) (/opt/firtool-1.139.0/bin/../lib/libMLIRPass.so+0x1a7a9)
#19 0x000055c4bee095d0 executeWithSources(mlir::MLIRContext*, llvm::SourceMgr&) circt-verilog.cpp:0:0
#20 0x000055c4bee04dd5 execute(mlir::MLIRContext*) circt-verilog.cpp:0:0
#21 0x000055c4bee044b8 main (/opt/firtool-1.139.0/bin/circt-verilog+0x84b8)
#22 0x00007fb02081c1ca __libc_start_call_main ./csu/../sysdeps/nptl/libc_start_call_main.h:74:3
#23 0x00007fb02081c28b call_init ./csu/../csu/libc-start.c:128:20
#24 0x00007fb02081c28b __libc_start_main ./csu/../csu/libc-start.c:347:5
#25 0x000055c4bee03b05 _start (/opt/firtool-1.139.0/bin/circt-verilog+0x7b05)
```

</details>

## Related Issues

This issue is related to but distinct from existing string type issues:

- **#8283**: [ImportVerilog] Cannot compile forward declared string type - focuses on string **variables**, not ports
- **#8332**: [MooreToCore] Support for StringType from moore to llvm dialect - feature request for string lowering design
- **#8930**: [MooreToCore] Crash with sqrt/floor - same assertion message but different trigger (real type conversion)

**Key Difference**: This bug specifically concerns `string` type **module ports** and crashes in `getModulePortInfo()` during `HWModuleOp` creation, which is a different code path from the variable handling in #8283 or the conversion operation in #8930.

---
*This issue was generated with assistance from an automated bug reporter.*
