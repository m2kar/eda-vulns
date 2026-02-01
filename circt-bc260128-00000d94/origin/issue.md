<!-- 
  CIRCT Bug Report
  Case ID: 260128-00000d94
  Generated: 2026-02-01
-->

## Title
[Moore] Crash in MooreToCore when converting string output port

## Description

CIRCT crashes with an assertion failure when processing a SystemVerilog module with a `string` type as an output port. The crash occurs during the MooreToCore conversion pass in the `SVModuleOpConversion::matchAndRewrite` method when attempting to convert the module's port information.

**Root Cause**: The type converter returns `null` (or an invalid type) for `StringType` when converting module ports, but the `getModulePortInfo()` function lacks null-checking before passing the result to `ModulePortInfo::sanitizeInOut()`. When `sanitizeInOut()` calls `dyn_cast<hw::InOutType>()` on a null type, LLVM's assertion `detail::isPresent(Val)` fails.

**Impact**: Valid SystemVerilog code causes an unhandled crash instead of producing a clear error message.

### Crash Details
- **Dialect**: Moore
- **Failing Pass**: convert-moore-to-core (MooreToCorePass)
- **Crash Type**: Assertion failure
- **CIRCT Version**: 1.139.0

## Steps to Reproduce

1. Save the test case below as `test.sv`
2. Run:
   ```bash
   /opt/firtool/bin/circt-verilog --ir-hw test.sv
   ```

## Expected Behavior

Either:
- Successfully compile the module (if string ports are supported), or
- Emit a clear error message (if string ports are unsupported for synthesis)

## Actual Behavior

```
circt-verilog: .../llvm/include/llvm/Support/Casting.h:650: 
decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]: 
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.

[Stack trace - see below]
```

Segmentation fault (exit code 139)

## Test Case

```systemverilog
module test_module(
  output string out_str
);
endmodule
```

## Error Output

```
PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
Stack dump:
0.	Program arguments: /opt/firtool/bin/circt-verilog --ir-hw bug.sv
 #0 0x00007f7ff02678a8 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/opt/firtool-1.139.0/bin/../lib/libLLVMSupport.so+0x2008a8)
 #1 0x00007f7ff02652f5 llvm::sys::RunSignalHandlers() (/opt/firtool-1.139.0/bin/../lib/libLLVMSupport.so+0x1fe2f5)
 #2 0x00007f7ff0268631 SignalHandler(int, siginfo_t*, void*) Signals.cpp:0:0
 #3 0x00007f7fefd75330 (/lib/x86_64-linux-gnu/libc.so.6+0x45330)
 #4 0x00007f7ff44898ae (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, circt::moore::SVModuleOpAdaptor, mlir::ConversionPatternRewriter&) const MooreToCore.cpp:0:0
 #5 0x00007f7ff4489b93 llvm::LogicalResult mlir::ConversionPattern::dispatchTo1To1<mlir::OpConversionPattern<circt::moore::SVModuleOp>, circt::moore::SVModuleOp>(mlir::OpConversionPattern<circt::moore::SVModuleOp> const&, circt::moore::SVModuleOp, circt::moore::SVModuleOp::GenericAdaptor<llvm::ArrayRef<mlir::ValueRange>>, mlir::ConversionPatternRewriter&) (/opt/firtool-1.139.0/bin/../lib/libCIRCTMooreToCore.so+0x50b93)
 [remaining frames...]
```

## Root Cause Analysis

### Hypothesis 1 (High Confidence): Missing Null Check

The `typeConverter.convertType()` call in `getModulePortInfo()` returns `null` for `StringType` (or a type that's invalid in the port context), but there is no validation before the type is stored in `PortInfo`.

**Evidence**:
- Assertion message: "dyn_cast on a non-existent value" → indicates null `mlir::Type`
- Test case uses `output string` port, which is uncommon and likely untested
- Stack trace shows crash in `sanitizeInOut()` which performs `dyn_cast<hw::InOutType>(p.type)` on all ports
- Code location: `lib/Conversion/MooreToCore/MooreToCore.cpp:243` calls `convertType()` without null check
- Code location: `include/circt/Dialect/HW/PortImplementation.h:177` crashes on `dyn_cast()` with null type

**Mechanism**:
```
1. MooreToCore conversion starts for the module
2. getModulePortInfo() iterates over module ports
3. For the string output port, typeConverter.convertType(port.type) returns null
4. The null type is passed to PortInfo constructor
5. ModulePortInfo constructor calls sanitizeInOut()
6. sanitizeInOut() iterates over ports and calls dyn_cast<InOutType>(null_type)
7. LLVM assertion "detail::isPresent(Val)" fails
```

### Hypothesis 2 (Medium Confidence): Invalid Type in Port Context

The `StringType` → `sim::DynamicStringType` conversion is registered, but `sim::DynamicStringType` is not a valid hardware port type and causes downstream failures.

**Evidence**:
- Type conversion is explicitly registered at `lib/Conversion/MooreToCore/MooreToCore.cpp:2304-2305`
- Dynamic strings have no natural hardware synthesis representation
- `hw::ModulePortInfo` may have constraints on port types that exclude dynamic types

## Environment

- **CIRCT Version**: firtool-1.139.0
- **LLVM Version**: 22.0.0git
- **Tool**: circt-verilog --ir-hw
- **OS**: Linux x86_64
- **Command**: `/opt/firtool/bin/circt-verilog --ir-hw bug.sv`

## Validation

The test case is valid IEEE 1800-2017 SystemVerilog:
- ✅ Accepted by slang: "Build succeeded: 0 errors, 0 warnings"
- ✅ Accepted by Verilator: No errors in lint-only mode
- ❌ Rejected by Icarus Verilog: "sorry: Port with type `string` is not supported"

**Conclusion**: The test case is valid SystemVerilog. Even if string ports are unsupported for synthesis, CIRCT should emit a clear diagnostic error, not crash.

## Stack Trace

<details>
<summary>Click to expand full stack trace</summary>

```
#0 0x00007f7ff02678a8 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int)
#1 0x00007f7ff02652f5 llvm::sys::RunSignalHandlers()
#2 0x00007f7ff0268631 SignalHandler(int, siginfo_t*, void*)
#3 0x00007f7fefd75330 (/lib/x86_64-linux-gnu/libc.so.6+0x45330)
#4 0x00007f7ff44898ae (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, circt::moore::SVModuleOpAdaptor, mlir::ConversionPatternRewriter&) const MooreToCore.cpp:0:0
#5 0x00007f7ff4489b93 llvm::LogicalResult mlir::ConversionPattern::dispatchTo1To1<mlir::OpConversionPattern<circt::moore::SVModuleOp>, circt::moore::SVModuleOp>(mlir::OpConversionPattern<circt::moore::SVModuleOp> const&, circt::moore::SVModuleOp, circt::moore::SVModuleOp::GenericAdaptor<llvm::ArrayRef<mlir::ValueRange>>, mlir::ConversionPatternRewriter&)
#6 0x00007f7ff4489530 mlir::OpConversionPattern<circt::moore::SVModuleOp>::matchAndRewrite(mlir::Operation*, llvm::ArrayRef<mlir::ValueRange>, mlir::ConversionPatternRewriter&) const
#7 0x00007f7ff279b438 mlir::ConversionPattern::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const
#8 0x00007f7ff27658ed void llvm::function_ref<void ()>::callback_fn<mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>)::$_0>(long)
#9 0x00007f7ff2762774 mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>)
#10 0x00007f7ff279cc6f (anonymous namespace)::OperationLegalizer::legalize(mlir::Operation*)
#11 0x00007f7ff279c470 mlir::OperationConverter::convert(mlir::Operation*, bool)
#12 0x00007f7ff279cdae mlir::OperationConverter::convertOperations(llvm::ArrayRef<mlir::Operation*>)
#13 0x00007f7ff27aa8e4 void llvm::function_ref<void ()>::callback_fn<applyConversion(llvm::ArrayRef<mlir::Operation*>, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig, (anonymous namespace)::OpConversionMode)::$_0>(long)
#14 0x00007f7ff27a1f7d applyConversion(llvm::ArrayRef<mlir::Operation*>, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig, (anonymous namespace)::OpConversionMode)
#15 0x00007f7ff27a20fe mlir::applyFullConversion(mlir::Operation*, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig)
#16 0x00007f7ff445b231 (anonymous namespace)::MooreToCorePass::runOnOperation()
#17 0x00007f7ff25572a5 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int)
#18 0x00007f7ff255a7a9 mlir::PassManager::run(mlir::Operation*)
#19 0x000056073fc335d0 executeWithSources(mlir::MLIRContext*, llvm::SourceMgr&) circt-verilog.cpp:0:0
#20 0x000056073fc2edd5 execute(mlir::MLIRContext*)
#21 0x000056073fc2e4b8 main (/opt/firtool/bin/circt-verilog+0x84b8)
#22 0x00007f7fefd5a1ca __libc_start_call_main
#23 0x00007f7fefd5a28b call_init
#24 0x00007f7fefd5a28b __libc_start_main
#25 0x000056073fc2db05 _start (/opt/firtool/bin/circt-verilog+0x7b05)
```

**Key frames**:
- `#4`: SVModuleOpConversion::matchAndRewrite (initial crash location)
- `#16`: MooreToCorePass::runOnOperation (conversion pass entry)
- `#17-18`: PassManager running the conversion

</details>

## Related Issues

This crash is part of a broader issue with `StringType` handling in MooreToCore conversion. Related issues:

- **#8930**: [MooreToCore] Crash with sqrt/floor - Similar assertion failure pattern (dyn_cast on non-existent value) but in a different context
- **#8332**: [MooreToCore] Support for StringType from moore to llvm dialect - Feature request for StringType lowering
- **#8283**: [ImportVerilog] Cannot compile forward declared string type - StringType handling in MooreToCore conversion

This issue is distinct from the above because it specifically targets **string type as module output port**, which may not be covered by existing reports.

## Suggested Fixes

1. **Immediate**: Add null check in `getModulePortInfo()` after `typeConverter.convertType()`:
   ```cpp
   Type portTy = typeConverter.convertType(port.type);
   if (!portTy) {
     // Emit a diagnostic and return failure
     return failure();
   }
   ```

2. **Defensive**: Add type validation in `sanitizeInOut()`:
   ```cpp
   void sanitizeInOut() {
     for (auto &p : ports)
       if (p.type && isa<hw::InOutType>(p.type)) {
         p.type = cast<hw::InOutType>(p.type).getElementType();
         p.dir = ModulePort::Direction::InOut;
       }
   }
   ```

3. **Better UX**: Emit a clear diagnostic when string ports are encountered:
   - "String type is not supported as module port for hardware synthesis"
   - Point user to the problematic port location

---

*This issue was generated by automated crash analysis. Case ID: 260128-00000d94*
