<!-- Title: [LLHD] Assertion failure in Mem2Reg when processing real type variables with clocked assignments -->

## Description

CIRCT's LLHD Mem2Reg pass crashes when processing SystemVerilog `real` (floating-point) type variables in clocked assignments (`always @(posedge clk)`).

The crash occurs in `Mem2RegPass` at `insertBlockArgs()` when attempting to create an MLIR `IntegerType` using `hw::getBitWidth()`. For unsupported types like `real`, `hw::getBitWidth()` returns -1, which is interpreted as an unsigned value (~2^64-1), causing an attempt to create an `IntegerType` exceeding MLIR's maximum bitwidth limit of 16,777,215 bits.

**Crash Type**: assertion
**Dialect**: LLHD
**Failing Pass**: Mem2Reg (lib/Dialect/LLHD/Transforms/Mem2Reg.cpp:1753)

## Steps to Reproduce

1. Save test case below as `test.sv`
2. Run:
   ```bash
   circt-verilog --ir-hw test.sv
   ```

## Test Case

```systemverilog
module test(
  input logic clk,
  input real in_real,
  output real out_real
);
  always @(posedge clk) begin
    out_real <= in_real;
  end
endmodule
```

## Error Output

```
<unknown>:0: error: integer bitwidth is limited to 16777215 bits
circt-verilog: /home/zhiqing/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/include/mlir/IR/StorageUniquerSupport.h:180: static ConcreteT mlir::detail::StorageUserBase<mlir::IntegerType, mlir::Type, mlir::detail::IntegerTypeStorage, mlir::detail::TypeUniquer, mlir::VectorElementTypeInterface::Trait>::get(MLIRContext *, Args &&...) [ConcreteT = mlir::IntegerType, BaseT = mlir::Type, StorageT = mlir::detail::IntegerTypeStorage, UniquerT = mlir::detail::TypeUniquer, Traits = <mlir::VectorElementTypeInterface::Trait>, Args = <unsigned int &, mlir::IntegerType::SignednessSemantics &>]: Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
...
```

## Root Cause Analysis

### Crash Path

1. `circt-verilog` processes SystemVerilog input with `--ir-hw` flag
2. The LLHD `Mem2RegPass::runOnOperation()` is invoked
3. `Promoter::promote()` is called to perform memory-to-register promotion
4. `Promoter::insertBlockArgs()` → `insertBlockArgs(BlockEntry*)` iterates block entries
5. At line 1742-1753 in `Mem2Reg.cpp`, when creating a default value for an uninitialized slot:
   ```cpp
   auto type = getStoredType(slot);
   auto flatType = builder.getIntegerType(hw::getBitWidth(type));
   ```
6. `hw::getBitWidth()` returns -1 for unsupported types (like `real`)
7. `-1` is interpreted as an unsigned integer, becoming an extremely large value
8. `builder.getIntegerType(-1)` tries to create an IntegerType with ~2^64-1 bits
9. MLIR's IntegerType verifier fails because width exceeds 16,777,215 bits

### Key Issue

The LLHD Mem2Reg pass calls `hw::getBitWidth()` without checking if the type is supported. When `getBitWidth()` returns -1 for `real` types, this value is used directly to create an `IntegerType`, causing the assertion failure.

### Validity

The test case is **valid SystemVerilog**:
- `real` type is a standard IEEE 1800 floating-point type
- Clocked assignment using `always @(posedge clk)` is standard

Cross-tool validation confirms:
- Verilator: ✅ No errors (lint-only mode)
- Slang: ✅ Build succeeded (0 errors, 0 warnings)

## Environment

- **CIRCT Version**: circt-1.139.0
- **OS**: Linux 5.15.0
- **Architecture**: x86_64

## Stack Trace

<details>
<summary>Click to expand stack trace</summary>

```
#0 0x000055b1e170732f llvm::sys::PrintStackTrace(llvm::raw_ostream&, int)
#1 0x000055b1e17082e9 llvm::sys::RunSignalHandlers()
#2 0x000055b1e17082e9 SignalHandler(int, siginfo_t*, void*)
#3 0x00007f7f4c301330 (/lib/x86_64-linux-gnu/libc.so.6+0x45330)
#4 0x00007f7f4c35ab2c __pthread_kill_implementation
#5 0x00007f7f4c35ab2c __pthread_kill_internal
#6 0x000055b1e05a9cea mlir::IntegerType::get(mlir::MLIRContext*, unsigned int, mlir::IntegerType::SignednessSemantics)
#7 0x000055b1dffcc429 (anonymous namespace)::Promoter::insertBlockArgs((anonymous namespace)::BlockEntry*)
#8 0x000055b1dffcc429 (anonymous namespace)::Promoter::insertBlockArgs()
#9 0x000055b1dffcc429 (anonymous namespace)::Promoter::promote()
#10 0x000055b1dffc4202 (anonymous namespace)::Mem2RegPass::runOnOperation()
#11 0x000055b1e13955a2 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int)::$_3::operator()()
#12 0x000055b1e13872b1 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int)
#13 0x000055b1e138827f mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*)
#14 0x000055b1e1397caa mlir::detail::OpToOpPassAdaptor::runOnOperationAsyncImpl(bool)::$_12::operator()()
```

</details>

## Related Issues

- #9287: [HW] Make `hw::getBitWidth` use std::optional vs -1
  - This issue addresses the root cause: `hw::getBitWidth()` returning -1 for unsupported types
  - The fix proposed in #9287 would prevent this specific crash
  - Consider adding this test case to #9287 as a concrete failing example

- #8693: [Mem2Reg] Local signal does not dominate final drive
- #8286: [circt-verilog][llhd][arcilator] Verilog-to-LLVM lowering issues

---
*This issue was generated with assistance from an automated bug reporter.*
