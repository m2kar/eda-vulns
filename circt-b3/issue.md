# [Moore] Assertion in MooreToCore when module uses packed union type as port

## Description

CIRCT crashes with assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"` when compiling SystemVerilog modules that use packed union types as module ports.

**Likely cause**: MooreToCore conversion pass lacks a type conversion rule for packed union types (`UnionType`). When processing module ports, the type converter fails to convert `UnionType`, resulting in an invalid/null type that causes assertion failures in downstream port processing code (`getModulePortInfo` at line 259).

> **Validation**: Test case accepted by: verilator, iverilog, slang
> **IEEE 1800-2005**: Packed unions are valid (Section 7.3)

## Steps to Reproduce

1. Save following code as `bug.sv`
2. Run: `circt-verilog --ir-hw bug.sv`

## Test Case

```systemverilog
typedef union packed {
  logic [31:0] a;
  logic [31:0] b;
} my_union;

module Sub(input my_union in_val, output my_union out_val);
  assign out_val = in_val;
endmodule

module Top;
  my_union data_in, data_out;
  
  Sub s(.in_val(data_in), .out_val(data_out));
endmodule
```

## Error Output

```
 #4 0x00007f8176f898ae (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, circt::moore::SVModuleOpAdaptor, mlir::ConversionPatternRewriter&) const MooreToCore.cpp:0:0
 #5 0x00007f8176f89b93 llvm::LogicalResult mlir::ConversionPattern::dispatchTo1To1<mlir::OpConversionPattern<circt::moore::SVModuleOp>, circt::moore::SVModuleOp>(mlir::OpConversionPattern<circt::moore::SVModuleOp> const&, circt::moore::SVModuleOp, circt::moore::SVModuleOp::GenericAdaptor<llvm::ArrayRef<mlir::ValueRange>>, mlir::ConversionPatternRewriter&) (/opt/firtool-1.139.0/bin/../lib/libCIRCTMooreToCore.so+0x50b93)
 #6 0x00007f8176f89530 mlir::OpConversionPattern<circt::moore::SVModuleOp>::matchAndRewrite(mlir::Operation*, llvm::ArrayRef<mlir::ValueRange>, mlir::ConversionPatternRewriter&) const (/opt/firtool-1.139.0/bin/../lib/libCIRCTMooreToCore.so+0x50530)
```

<details>
<summary>Full Stack Trace</summary>

```
PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
Stack dump:
0.      Program arguments: /opt/firtool/bin/circt-verilog --ir-hw bug.sv
 #0 0x00007f8172d678a8 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/opt/firtool-1.139.0/bin/../lib/libLLVMSupport.so+0x2008a8)
 #1 0x00007f8172d652f5 llvm::sys::RunSignalHandlers() (/opt/firtool-1.139.0/bin/../lib/libLLVMSupport.so+0x1fe2f5)
 #2 0x00007f8172d68631 SignalHandler(int, siginfo_t*, void*) Signals.cpp:0:0
 #3 0x00007f8172875330 (/lib/x86_64-linux-gnu/libc.so.6+0x45330)
 #4 0x00007f8176f898ae (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, circt::moore::SVModuleOpAdaptor, mlir::ConversionPatternRewriter&) const MooreToCore.cpp:0:0
 #5 0x00007f8176f89b93 llvm::LogicalResult mlir::ConversionPattern::dispatchTo1To1<mlir::OpConversionPattern<circt::moore::SVModuleOp>, circt::moore::SVModuleOp>(mlir::OpConversionPattern<circt::moore::SVModuleOp> const&, circt::moore::SVModuleOp, circt::moore::SVModuleOp::GenericAdaptor<llvm::ArrayRef<mlir::ValueRange>>, mlir::ConversionPatternRewriter&) (/opt/firtool-1.139.0/bin/../lib/libCIRCTMooreToCore.so+0x50b93)
 #6 0x00007f8176f89530 mlir::OpConversionPattern<circt::moore::SVModuleOp>::matchAndRewrite(mlir::Operation*, llvm::ArrayRef<mlir::ValueRange>, mlir::ConversionPatternRewriter&) const (/opt/firtool-1.139.0/bin/../lib/libCIRCTMooreToCore.so+0x50530)
 #7 0x00007f817529b438 mlir::ConversionPattern::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const (/opt/firtool-1.139.0/bin/../lib/libMLIRTransformUtils.so+0x2a438)
 #8 0x00007f81752658ed void llvm::function_ref<void ()>::callback_fn<mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>)::$_0>(long) PatternApplicator.cpp:0:0
 #9 0x00007f8175262774 mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>) (/opt/firtool-1.139.0/bin/../lib/libMLIRRewrite.so+0x7774)
#10 0x00007f817529cc6f (anonymous namespace)::OperationLegalizer::legalize(mlir::Operation*) DialectConversion.cpp:0:0
#11 0x00007f817529c470 mlir::OperationConverter::convert(mlir::Operation*, bool) (/opt/firtool-1.139.0/bin/../lib/libMLIRTransformUtils.so+0x2b470)
#12 0x00007f817529cdae mlir::OperationConverter::convertOperations(llvm::ArrayRef<mlir::Operation*>) (/opt/firtool-1.139.0/bin/../lib/libMLIRTransformUtils.so+0x2bdae)
#13 0x00007f81752aa8e4 void llvm::function_ref<void ()>::callback_fn<applyConversion(llvm::ArrayRef<mlir::Operation*>, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig, (anonymous namespace)::OpConversionMode)::$_0>(long) DialectConversion.cpp:0:0
#14 0x00007f81752a1f7d applyConversion(llvm::ArrayRef<mlir::Operation*>, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig, (anonymous namespace)::OpConversionMode) DialectConversion.cpp:0:0
#15 0x00007f81752a20fe mlir::applyFullConversion(mlir::Operation*, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig) (/opt/firtool-1.139.0/bin/../lib/libMLIRTransformUtils.so+0x310fe)
#16 0x00007f8176f5b231 (anonymous namespace)::MooreToCorePass::runOnOperation() MooreToCore.cpp:0:0
#17 0x00007f81750572a5 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) (/opt/firtool-1.139.0/bin/../lib/libMLIRPass.so+0x172a5)
#18 0x00007f817505a7a9 mlir::PassManager::run(mlir::Operation*) (/opt/firtool-1.139.0/bin/../lib/libMLIRPass.so+0x1a7a9)
#19 0x0000563df3daf5d0 executeWithSources(mlir::MLIRContext*, llvm::SourceMgr&) circt-verilog.cpp:0:0
#20 0x0000563df3daadd5 execute(mlir::MLIRContext*) circt-verilog.cpp:0:0
#21 0x0000563df3daa4b8 main (/opt/firtool/bin/circt-verilog+0x84b8)
#22 0x00007f817285a1ca __libc_start_call_main ./csu/../sysdeps/nptl/libc_start_call_main.h:74:3
#23 0x00007f817285a28b call_init ./csu/../csu/libc-start.c:128:20
#24 0x00007f817285a28b __libc_start_main ./csu/../csu/libc-start.c:347:5
#25 0x0000563df3da9b05 _start (/opt/firtool/bin/circt-verilog+0x7b05)
[1]    1143009 segmentation fault (core dumped)  /opt/firtool/bin/circt-verilog --ir-hw bug.sv

```

</details>

## Root Cause Analysis

- **Dialect**: Moore
- **Failing Pass**: MooreToCore

### Hypotheses

1. **Missing UnionType type conversion rule** (confidence: high)
   - Evidence: ["Test case uses 'typedef union packed' as module port type", 'Stack trace shows crash in getModulePortInfo during port processing at line 259', "Assertion message indicates 'dyn_cast<InOutType>' failed on non-existent value", 'No conversion rule for UnionType exists in populateTypeConversion function (lines 2268-2409)', 'Similar types like StructType have conversion rules, but UnionType does not', 'Both UnionType and StructType implement DestructurableTypeInterface, suggesting they should be handled similarly']
   - Mechanism: The typeConverter fails to convert UnionType when processing module ports, resulting in an invalid type that causes assertion failures

2. **Type converter returns invalid non-null type** (confidence: medium)
   - Evidence: ["Line 245-248 checks 'if (!portTy)' and emits error", 'However, crash still occurs, suggesting either the check is not being reached or the type is non-null but invalid', 'The assertion occurs in a different code path that may not be covered by the initial null check']
   - Mechanism: The typeConverter may return a non-null type with incorrect MLIR type ID or metadata

## Environment

- **CIRCT Version**: LLVM (http://llvm.org/):
  LLVM version 22.0.0git
  Optimized build.
CIRCT firtool-1.139.0
slang version 9.1.0+0

---
**Labels**: bug, Moore
