# [Moore] Crash when module has packed union type port

## Description

CIRCT crashes with an internal compiler error when compiling SystemVerilog code that uses packed unions as module ports. The crash occurs during the MooreToCore pass conversion when attempting to convert UnionType ports to HW dialect types.

**Likely cause**: Missing UnionType type conversion in MooreToCore pass


> **Validation**: Test case accepted by: iverilog

## Steps to Reproduce

1. Save following code as `test.sv`
2. Run: `circt-verilog --ir-hw test.sv`

## Test Case

```systemverilog
typedef union packed {
  logic [31:0] a;
} my_union;

module mod1(output my_union out);
endmodule

```

## Error Output

```
PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
```

## Root Cause Analysis

- **Dialect**: Moore
- **Failing Pass**: MooreToCore
- **Crash Category**: Missing type conversion

### Hypotheses

1. **Missing UnionType type conversion** (confidence: high)
   - Evidence:
     - Test case uses packed union (UnionType) as module port type
     - Stack trace shows crash in getModulePortInfo() during port type conversion
     - Code review reveals no addConversion for UnionType/UnpackedUnionType in populateTypeConversion()
     - Similar types (StructType, UnpackedStructType) have conversions defined
     - Assertion 'dyn_cast on a non-existent value' consistent with null type from failed conversion
   - Mechanism: When a module has a UnionType port, typeConverter.convertType() returns null because no conversion is registered. Subsequent dyn_cast<InOutType> on the null type triggers the assertion.

2. **Incorrect port direction handling** (confidence: low)
   - Evidence:
     - Assertion involves InOutType suggesting attempt to cast port type
     - Code has FIXME about not supporting inout/ref ports
     - But test case uses explicit input/output, not inout

## Environment

- **CIRCT Version**: LLVM (http://llvm.org/):
  LLVM version 22.0.0git
  Optimized build.
CIRCT firtool-1.139.0
slang version 9.1.0+0


<details>
<summary>Stack Trace</summary>

```
0.	Program arguments: /opt/firtool/bin/circt-verilog --ir-hw bug.sv
 #0 0x00007f1e2902d8a8 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/opt/firtool-1.139.0/bin/../lib/libLLVMSupport.so+0x2008a8)
 #1 0x00007f1e2902b2f5 llvm::sys::RunSignalHandlers() (/opt/firtool-1.139.0/bin/../lib/libLLVMSupport.so+0x1fe2f5)
 #2 0x00007f1e2902e631 SignalHandler(int, siginfo_t*, void*) Signals.cpp:0:0
 #3 0x00007f1e28b3b330 (/lib/x86_64-linux-gnu/libc.so.6+0x45330)
 #4 0x00007f1e2d24f8ae (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, circt::moore::SVModuleOpAdaptor, mlir::ConversionPatternRewriter&) const MooreToCore.cpp:0:0
 #5 0x00007f1e2d24fb93 llvm::LogicalResult mlir::ConversionPattern::dispatchTo1To1<mlir::OpConversionPattern<circt::moore::SVModuleOp>, circt::moore::SVModuleOp>(mlir::OpConversionPattern<circt::moore::SVModuleOp> const&, circt::moore::SVModuleOp, circt::moore::SVModuleOp::GenericAdaptor<llvm::ArrayRef<mlir::ValueRange>>, mlir::ConversionPatternRewriter&) (/opt/firtool-1.139.0/bin/../lib/libCIRCTMooreToCore.so+0x50b93)
 #6 0x00007f1e2d24f530 mlir::OpConversionPattern<circt::moore::SVModuleOp>::matchAndRewrite(mlir::Operation*, llvm::ArrayRef<mlir::ValueRange>, mlir::ConversionPatternRewriter&) const (/opt/firtool-1.139.0/bin/../lib/libCIRCTMooreToCore.so+0x50530)
 #7 0x00007f1e2b561438 mlir::ConversionPattern::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const (/opt/firtool-1.139.0/bin/../lib/libMLIRTransformUtils.so+0x2a438)
 #8 0x00007f1e2b52b8ed void llvm::function_ref<void ()>::callback_fn<mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>)::$_0>(long) PatternApplicator.cpp:0:0
 #9 0x00007f1e2b528774 mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>) (/opt/firtool-1.139.0/bin/../lib/libMLIRRewrite.so+0x7774)
#10 0x00007f1e2b562c6f (anonymous namespace)::OperationLegalizer::legalize(mlir::Operation*) DialectConversion.cpp:0:0
#11 0x00007f1e2b562470 mlir::OperationConverter::convert(mlir::Operation*, bool) (/opt/firtool-1.139.0/bin/../lib/libMLIRTransformUtils.so+0x2b470)
#12 0x00007f1e2b562dae mlir::OperationConverter::convertOperations(llvm::ArrayRef<mlir::Operation*>) (/opt/firtool-1.139.0/bin/../lib/libMLIRTransformUtils.so+0x2bdae)
#13 0x00007f1e2b5708e4 void llvm::function_ref<void ()>::callback_fn<applyConversion(llvm::ArrayRef<mlir::Operation*>, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig, (anonymous namespace)::OpConversionMode)::$_0>(long) DialectConversion.cpp:0:0
#14 0x00007f1e2b567f7d applyConversion(llvm::ArrayRef<mlir::Operation*>, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig, (anonymous namespace)::OpConversionMode) DialectConversion.cpp:0:0
#15 0x00007f1e2b5680fe mlir::applyFullConversion(mlir::Operation*, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig) (/opt/firtool-1.139.0/bin/../lib/libMLIRTransformUtils.so+0x310fe)
#16 0x00007f1e2d221231 (anonymous namespace)::MooreToCorePass::runOnOperation() MooreToCore.cpp:0:0
#17 0x00007f1e2b31d2a5 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) (/opt/firtool-1.139.0/bin/../lib/libMLIRPass.so+0x172a5)
#18 0x00007f1e2b3207a9 mlir::PassManager::run(mlir::Operation*) (/opt/firtool-1.139.0/bin/../lib/libMLIRPass.so+0x1a7a9)
#19 0x000055de77eb75d0 executeWithSources(mlir::MLIRContext*, llvm::SourceMgr&) circt-verilog.cpp:0:0
#20 0x000055de77eb2dd5 execute(mlir::MLIRContext*) circt-verilog.cpp:0:0
#21 0x000055de77eb24b8 main (/opt/firtool/bin/circt-verilog+0x84b8)
#22 0x00007f1e28b201ca __libc_start_call_main ./csu/../sysdeps/nptl/libc_start_call_main.h:74:3
#23 0x00007f1e28b2028b call_init ./csu/../csu/libc-start.c:128:20
#24 0x00007f1e28b2028b __libc_start_main ./csu/../csu/libc-start.c:347:5
#25 0x000055de77eb1b05 _start (/opt/firtool/bin/circt-verilog+0x7b05)
```

</details>


---
**Labels**: bug, Moore
