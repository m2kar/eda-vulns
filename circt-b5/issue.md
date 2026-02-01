<!-- Title: [Comb] Assertion failure in Canonicalizer with extractConcatToConcatExtract pattern -->

## Description

CIRCT crashes with an assertion failure when processing SystemVerilog code that uses mixed continuous and procedural assignments to different bits of the same output signal. The crash occurs during the Canonicalizer pass when the `extractConcatToConcatExtract` fold pattern attempts to replace an `ExtractOp` with a single value.

**Root Cause**: The `extractConcatToConcatExtract` pattern in `lib/Dialect/Comb/CombFolds.cpp:547` calls `replaceOpAndCopyNamehint()` which invokes `replaceOp()`, but the operation unexpectedly still has uses when `eraseOp()` is called. This occurs because the GreedyPatternRewriteDriver may apply multiple canonicalization patterns concurrently or in rapid succession, and the pattern logic doesn't properly account for operations that may have already been modified by other patterns in the same rewrite iteration.

**Crash Type**: assertion
**Dialect**: Comb
**Failing Pass**: Canonicalizer (GreedyPatternRewriteDriver)

## Steps to Reproduce

1. Save the test case below as `test.sv`
2. Run:
   ```bash
   circt-verilog --ir-hw test.sv
   ```

## Test Case

```systemverilog
module test_module(
  input  logic [1:0] in,
  output logic [3:0] out
);

  // Continuous assignments to output bits
  assign out[0] = in[0] ^ in[1];
  assign out[3] = 1'h0;

  // Combinational always block with implicit sensitivity list
  always @* begin
    out[1] = in[0] & in[1];
    out[2] = in[0] | in[1];
  end

endmodule
```

## Error Output

```
PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
Stack dump:
0.      Program arguments: circt-verilog --ir-hw bug.sv
 #0 0x00007f1d3e6bf8a8 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) (/opt/firtool-1.139.0/bin/../lib/libLLVMSupport.so+0x2008a8)
 #1 0x00007f1d3e6bd2f5 llvm::sys::RunSignalHandlers() (/opt/firtool-1.139.0/bin/../lib/libLLVMSupport.so+0x1fe2f5)
 #2 0x00007f1d3e6c0631 SignalHandler(int, siginfo_t*, void*) Signals.cpp:0:0
 #3 0x00007f1d3e1cd330 (/lib/x86_64-linux-gnu/libc.so.6+0x45330)
 #4 0x00007f1d40f55713 circt::comb::ExtractOp::canonicalize(circt::comb::ExtractOp, mlir::PatternRewriter&) (/opt/firtool-1.139.0/bin/../lib/libCIRCTComb.so+0x41713)
 #5 0x00007f1d40bbd8ed void llvm::function_ref<void ()>::callback_fn<mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>)::$_0>(long) PatternApplicator.cpp:0:0
 #6 0x00007f1d40bba774 mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>) (/opt/firtool-1.139.0/bin/../lib/libMLIRRewrite.so+0x7774)
 #7 0x00007f1d40c124a7 (anonymous namespace)::GreedyPatternRewriteDriver::processWorklist() GreedyPatternRewriteDriver.cpp:0:0
 #8 0x00007f1d40c0ffd9 mlir::applyPatternsGreedily(mlir::Region&, mlir::FrozenRewritePatternSet const&, mlir::GreedyRewriteConfig, bool*) (/opt/firtool-1.139.0/bin/../lib/libMLIRTransformUtils.so+0x46fd9)
 #9 0x00007f1d40c55d35 (anonymous namespace)::Canonicalizer::runOnOperation() Canonicalizer.cpp:0:0
#10 0x00007f1d409af2a5 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) (/opt/firtool-1.139.0/bin/../lib/libMLIRPass.so+0x172a5)
#11 0x00007f1d409aff48 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) (/opt/firtool-1.139.0/bin/../lib/libMLIRPass.so+0x17f48)
#12 0x00007f1d409b1663 mlir::detail::OpToOpPassAdaptor::runOnOperationAsyncImpl(bool) (/opt/firtool-1.139.0/bin/../lib/libMLIRPass.so+0x19663)
#13 0x00007f1d409af7e5 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) (/opt/firtool-1.139.0/bin/../lib/libMLIRPass.so+0x177e5)
#14 0x00007f1d409b27a9 mlir::PassManager::run(mlir::Operation*) (/opt/firtool-1.139.0/bin/../lib/libMLIRPass.so+0x1a7a9)
#15 0x000055df3c65a5d0 executeWithSources(mlir::MLIRContext*, llvm::SourceMgr&) circt-verilog.cpp:0:0
#16 0x000055df3c655dd5 execute(mlir::MLIRContext*) circt-verilog.cpp:0:0
#17 0x000055df3c6554b8 main (/opt/firtool-1.139.0/bin/circt-verilog+0x84b8)
#18 0x00007f1d3e1b21ca __libc_start_call_main ./csu/../sysdeps/nptl/libc_start_call_main.h:74:3
#19 0x00007f1d3e1b228b call_init ./csu/../csu/libc-start.c:128:20
#20 0x00007f1d3e1b228b __libc_start_main ./csu/../csu/libc-start.c:347:5
#21 0x000055df3c654b05 _start (/opt/firtool-1.139.0/bin/circt-verilog+0x7b05)
[1]    1198938 segmentation fault (core dumped)  circt-verilog --ir-hw bug.sv
```

## Root Cause Analysis

The crash occurs in the `extractConcatToConcatExtract` canonicalization pattern when it attempts to simplify an `extract(concat(...))` operation to a single value. The problematic code path is:

**File**: `lib/Dialect/Comb/CombFolds.cpp:547`

```cpp
if (reverseConcatArgs.size() == 1) {
  replaceOpAndCopyNamehint(rewriter, op, reverseConcatArgs[0]);  // CRASH HERE
}
```

**Call Stack**:
```
ExtractOp::canonicalize
  → extractConcatToConcatExtract
    → replaceOpAndCopyNamehint
      → mlir::RewriterBase::replaceOp
        → mlir::RewriterBase::eraseOp [ASSERTION FAILURE]
```

**Hypothesis (High Confidence)**: The GreedyPatternRewriteDriver applies multiple canonicalization patterns in rapid succession. When `reverseConcatArgs.size() == 1`, the pattern calls `replaceOpAndCopyNamehint()` which calls `replaceOp()`. The `replaceOp()` function calls `replaceAllOpUsesWith()` followed by `eraseOp()`. However, if another pattern has already modified the operation's uses in the same worklist iteration, the assertion `op->use_empty()` fails.

**Test Case Pattern**: The test case uses mixed continuous assignments (`assign`) and procedural assignments (`always @*`) to different bits of the same 4-bit output signal. This creates multiple extract operations from the same source, which triggers the optimization bug when the IR is lowered to Comb dialect and canonicalized.

## Environment

- **CIRCT Version**: firtool-1.139.0
- **LLVM Version**: 22.0.0git
- **OS**: Linux
- **Architecture**: x86_64

## Suggested Fixes

1. **Add use checking before replaceOp**: Verify that the operation is in a valid state before calling `replaceOpAndCopyNamehint`
2. **Return failure() on edge cases**: Be more defensive and return `failure()` if encountering situations that might lead to invalid IR state
3. **Improve rewriter API usage**: Use proper rewriter API sequence instead of relying on `replaceOp` to handle namehint and replacement separately

## Related Information

- **Affected Operations**: `comb.extract`, `comb.concat`
- **Affected Files**: `lib/Dialect/Comb/CombFolds.cpp`, `lib/Support/Naming.cpp`
- **Keywords**: canonicalizer, assertion, use_empty, ExtractOp, extractConcatToConcatExtract, GreedyPatternRewriteDriver

