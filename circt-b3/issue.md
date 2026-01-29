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
Crash Type: assertion
Hash: 03ce98b35955
Original Program File: program_20260125_204539_113699.sv

--- Compilation Command ---
/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw /tmp/featurefuzz_sv_85ldhy3n/test_03ce98b35955.sv | /edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/arcilator | /edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/opt -O0 | /edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/llc -O0 --filetype=obj -o /tmp/featurefuzz_sv_85ldhy3n/test_03ce98b35955.o

--- Error Message ---
circt-verilog: /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/Casting.h:650: decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]: Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
```

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

3. **Packed unions require special HW dialect representation** (confidence: low)
   - Evidence: ['HW dialect has hw::StructType but no explicit hw::UnionType', 'Packed unions in SystemVerilog have special semantics (overlapping storage)', 'May need to be represented differently in HW dialect as arrays or struct types']
   - Mechanism: Union types may need to be lowered to struct types or other representation

## Environment

- **CIRCT Version**: LLVM (http://llvm.org/):
  LLVM version 22.0.0git
  Optimized build.
CIRCT firtool-1.139.0
slang version 9.1.0+0


<details>
<summary>Stack Trace</summary>

```
0.	Program arguments: /edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw /tmp/featurefuzz_sv_85ldhy3n/test_03ce98b35955.sv
 #0 0x000056008ca49ce7 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/lib/Support/Unix/Signals.inc:842:13
 #1 0x000056008ca460e2 llvm::sys::RunSignalHandlers() /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/lib/Support/Signals.cpp:0:5
 #2 0x000056008ca4aacd SignalHandler(int, siginfo_t*, void*) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/lib/Support/Unix/Signals.inc:429:38
 #3 0x00007fe1051b3330 (/lib/x86_64-linux-gnu/libc.so.6+0x45330)
 #4 0x00007fe10520cb2c __pthread_kill_implementation ./nptl/pthread_kill.c:44:76
 #5 0x00007fe10520cb2c __pthread_kill_internal ./nptl/pthread_kill.c:78:10
 #6 0x00007fe10520cb2c pthread_kill ./nptl/pthread_kill.c:89:10
 #7 0x00007fe1051b327e raise ./signal/../sysdeps/posix/raise.c:27:6
 #8 0x00007fe1051968ff abort ./stdlib/abort.c:81:7
 #9 0x00007fe10519681b _nl_load_domain ./intl/loadmsgcat.c:1177:9
#10 0x00007fe1051a9517 (/lib/x86_64-linux-gnu/libc.so.6+0x3b517)
#11 0x000056008aa02b57 (/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog+0x1dbcb57)
#12 0x000056008b00b717 llvm::SmallVector<circt::hw::PortInfo, 1u>::~SmallVector() /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/ADT/SmallVector.h:1207:18
#13 0x000056008b00b717 (anonymous namespace)::getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Conversion/MooreToCore/MooreToCore.cpp:259:1
#14 0x000056008b00b717 (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, circt::moore::SVModuleOpAdaptor, mlir::ConversionPatternRewriter&) const /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Conversion/MooreToCore/MooreToCore.cpp:276:32
#15 0x000056008b00bb82 llvm::LogicalResult mlir::ConversionPattern::dispatchTo1To1<mlir::OpConversionPattern<circt::moore::SVModuleOp>, circt::moore::SVModuleOp>(mlir::OpConversionPattern<circt::moore::SVModuleOp> const&, circt::moore::SVModuleOp, circt::moore::SVModuleOp::GenericAdaptor<llvm::ArrayRef<mlir::ValueRange>>, mlir::ConversionPatternRewriter&) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/Transforms/DialectConversion.h:1069:15
#16 0x000056008b00b405 mlir::OpConversionPattern<circt::moore::SVModuleOp>::matchAndRewrite(mlir::Operation*, llvm::ArrayRef<mlir::ValueRange>, mlir::ConversionPatternRewriter&) const /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/Transforms/DialectConversion.h:715:3
#17 0x000056008c46b46a llvm::SmallVector<mlir::ValueRange, 3u>::~SmallVector() /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/ADT/SmallVector.h:1207:18
#18 0x000056008c46b46a mlir::ConversionPattern::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Transforms/Utils/DialectConversion.cpp:2404:1
#19 0x000056008c4d3816 mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>)::$_0::operator()() const /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Rewrite/PatternApplicator.cpp:223:13
#20 0x000056008c4d3816 void llvm::function_ref<void ()>::callback_fn<mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>)::$_0>(long) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/ADT/STLFunctionalExtras.h:46:12
#21 0x000056008c4cc7ab mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Rewrite/PatternApplicator.cpp:242:9
#22 0x000056008c46c33d llvm::succeeded(llvm::LogicalResult) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/LogicalResult.h:67:45
#23 0x000056008c46c33d (anonymous namespace)::OperationLegalizer::legalize(mlir::Operation*) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Transforms/Utils/DialectConversion.cpp:2602:7
#24 0x000056008c46b600 llvm::failed(llvm::LogicalResult) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/LogicalResult.h:71:42
#25 0x000056008c46b600 mlir::OperationConverter::convert(mlir::Operation*, bool) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Transforms/Utils/DialectConversion.cpp:3300:7
#26 0x000056008c46c6ca llvm::failed(llvm::LogicalResult) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/LogicalResult.h:71:42
#27 0x000056008c46c6ca llvm::LogicalResult mlir::OperationConverter::legalizeOperations<mlir::OperationConverter::applyConversion(llvm::ArrayRef<mlir::Operation*>)::$_0>(llvm::ArrayRef<mlir::Operation*>, mlir::OperationConverter::applyConversion(llvm::ArrayRef<mlir::Operation*>)::$_0, bool) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Transforms/Utils/DialectConversion.cpp:3401:9
#28 0x000056008c46c6ca mlir::OperationConverter::applyConversion(llvm::ArrayRef<mlir::Operation*>) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Transforms/Utils/DialectConversion.cpp:3447:26
#29 0x000056008c485156 applyConversion(llvm::ArrayRef<mlir::Operation*>, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig, (anonymous namespace)::OpConversionMode)::$_0::operator()() const /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Transforms/Utils/DialectConversion.cpp:4170:9
#30 0x000056008c485156 void llvm::function_ref<void ()>::callback_fn<applyConversion(llvm::ArrayRef<mlir::Operation*>, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig, (anonymous namespace)::OpConversionMode)::$_0>(long) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/ADT/STLFunctionalExtras.h:46:12
#31 0x000056008c47454e llvm::SmallVector<mlir::IRUnit, 6u>::~SmallVector() /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/ADT/SmallVector.h:1207:18
#32 0x000056008c47454e applyConversion(llvm::ArrayRef<mlir::Operation*>, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig, (anonymous namespace)::OpConversionMode) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Transforms/Utils/DialectConversion.cpp:4174:1
#33 0x000056008c4746e3 mlir::applyFullConversion(mlir::Operation*, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Transforms/Utils/DialectConversion.cpp:4207:3
#34 0x000056008af97b00 llvm::failed(llvm::LogicalResult) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/LogicalResult.h:71:42
#35 0x000056008af97b00 (anonymous namespace)::MooreToCorePass::runOnOperation() /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Conversion/MooreToCore/MooreToCore.cpp:2571:7
#36 0x000056008c68c3d6 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int)::$_3::operator()() const /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Pass/Pass.cpp:613:24
#37 0x000056008c68c3d6 void llvm::function_ref<void ()>::callback_fn<mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int)::$_3>(long) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/ADT/STLFunctionalExtras.h:46:12
#38 0x000056008c67c13f mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Pass/Pass.cpp:619:17
#39 0x000056008c67d107 llvm::failed(llvm::LogicalResult) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/LogicalResult.h:71:42
#40 0x000056008c67d107 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Pass/Pass.cpp:688:9
#41 0x000056008c686118 mlir::PassManager::runPasses(mlir::Operation*, mlir::AnalysisManager) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Pass/Pass.cpp:1123:3
#42 0x000056008c685891 mlir::PassManager::run(mlir::Operation*) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Pass/Pass.cpp:1097:0
#43 0x000056008a8a4ef9 llvm::failed(llvm::LogicalResult) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/LogicalResult.h:71:42
#44 0x000056008a8a4ef9 executeWithSources(mlir::MLIRContext*, llvm::SourceMgr&) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/tools/circt-verilog/circt-verilog.cpp:398:9
#45 0x000056008a898bd0 execute(mlir::MLIRContext*) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/tools/circt-verilog/circt-verilog.cpp:475:10
#46 0x000056008a89689b main /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/tools/circt-verilog/circt-verilog.cpp:528:8
#47 0x00007fe1051981ca __libc_start_call_main ./csu/../sysdeps/nptl/libc_start_call_main.h:74:3
#48 0x00007fe10519828b call_init ./csu/../csu/libc-start.c:128:20
#49 0x00007fe10519828b __libc_start_main ./csu/../csu/libc-start.c:347:5
#50 0x000056008a895f35 _start (/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog+0x1c4ff35)
```

</details>


---
**Labels**: bug, Moore
