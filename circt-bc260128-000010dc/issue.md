# arcilator assertion failure in InferStatePropertiesPass with array-typed state variables

## Description

`arcilator` crashes with an assertion failure when processing SystemVerilog code that contains array-typed state variables in `always_ff` blocks. The crash occurs in `InferStateProperties.cpp` during the `applyEnableTransformation` function when it attempts to create a `hw::ConstantOp` for enable signals without validating that the argument type is compatible (integer type required).

The assertion `isa<To>(Val) && "cast<Ty>() argument of incompatible type!"'` fails because `hw::ConstantOp::create` requires an `mlir::IntegerType`, but the code is being called with `hw::ArrayType` from array-typed state variables.

## Reproduction Steps

1. Create a SystemVerilog module with:
   - Array-typed state variables (both packed and unpacked arrays)
   - An `always_ff` block with conditional assignment
   - A `for` loop with array assignments
2. Run through the CIRCT pipeline:
   ```bash
   circt-verilog --ir-hw bug.sv | arcilator
   ```

## Minimal Testcase

```systemverilog
module test_module(
    input logic clk,
    input logic [7:0] data_in,
    output logic [7:0] out_unpacked,
    output logic [7:0] out_packed
);
  logic [7:0] packed_arr [0:1];
  logic [7:0] unpacked_arr [0:1];

  always_ff @(posedge clk) begin
    if (data_in == 0)
      unpacked_arr[0] <= 8'hFF;
    else
      unpacked_arr[0] <= data_in;

    for (int i = 1; i < 2; i++)
      unpacked_arr[i] <= data_in;

    packed_arr[0] <= data_in;
  end

  assign out_unpacked = unpacked_arr[0];
  assign out_packed = packed_arr[0];

endmodule
```

## Expected Behavior

The code should compile without crashes. The testcase uses standard SystemVerilog constructs:
- Unpacked arrays (`logic [7:0] arr [0:1]`)
- Sequential logic blocks (`always_ff @(posedge clk)`)
- For loops with array assignments
- Conditional assignments (if-else)

All syntax validators (slang, verilator, circt-verilog) accept this code.

## Actual Behavior

`arcilator` crashes with an assertion failure:

```
arcilator: /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/Casting.h:566: decltype(auto) llvm::cast(From &) [To = mlir::IntegerType, From = mlir::Type]: Assertion `isa<To>(Val) && "cast<Ty>() argument of incompatible type!"' failed.
```

## Stack Trace

```
#0  llvm::sys::PrintStackTrace(llvm::raw_ostream&, int)
#1  llvm::sys::RunSignalHandlers()
#2  SignalHandler(int, siginfo_t*, void*)
#3  (in libc.so.6)
#4  __pthread_kill_implementation
#5  pthread_kill
#6  raise
#7  abort
#8  (in arcilator)
#9  circt::hw::ConstantOp::create(mlir::OpBuilder&, mlir::Location, mlir::Type, long)
    /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/build/tools/circt/include/circt/Dialect/HW/HW.cpp.inc:2591:55
#10 mlir::Operation::getOpResultImpl(unsigned int)
#11 mlir::Operation::getResult(unsigned int)
#12 mlir::OpTrait::OneTypedResult<...>::Impl<circt::hw::ConstantOp>::getResult()
#13 mlir::OpTrait::OneTypedResult<...>::Impl<circt::hw::ConstantOp>::operator mlir::detail::TypedValue<...>()
#14 applyEnableTransformation(circt::arc::DefineOp, circt::arc::StateOp, llvm::ArrayRef<(anonymous namespace)::EnableInfo>)
    /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Dialect/Arc/Transforms/InferStateProperties.cpp:211:55
#15 (anonymous namespace)::InferStatePropertiesPass::runOnStateOp(circt::arc::StateOp, circt::arc::DefineOp, llvm::DenseMap<...>&)
    /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Dialect/Arc/Transforms/InferStateProperties.cpp:454:17
#16 mlir::detail::walk<...>(mlir::Operation*, llvm::function_ref<void (mlir::Operation*)>, mlir::WalkOrder)
#17 mlir::detail::walk<...>(mlir::Operation*, llvm::function_ref<void (mlir::Operation*)>, mlir::WalkOrder)
#18 (anonymous namespace)::InferStatePropertiesPass::runOnOperation()
    /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Dialect/Arc/Transforms/InferStateProperties.cpp:401:1
#19 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int)
#20 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*)
#21 mlir::PassManager::runPasses(mlir::Operation*, mlir::AnalysisManager)
#22 mlir::PassManager::run(mlir::Operation*)
#23 processBuffer(mlir::MLIRContext&, mlir::TimingScope&, llvm::SourceMgr&, std::optional<std::unique_ptr<llvm::ToolOutputFile, std::default_delete<llvm::ToolOutputFile>>>&)
    /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/tools/arcilator/arcilator.cpp:330:7
#24 processInputSplit(mlir::MLIRContext&, mlir::TimingScope&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, std::optional<std::unique_ptr<llvm::ToolOutputFile, std::default_delete<llvm::ToolOutputFile>>>&)
    /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/tools/arcilator/arcilator.cpp:546:12
#25 processInput(mlir::MLIRContext&, mlir::TimingScope&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, std::optional<std::unique_ptr<llvm::ToolOutputFile, std::default_delete<llvm::ToolOutputFile>>>&)
    /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/tools/arcilator/arcilator.cpp:562:12
#26 executeArcilator(mlir::MLIRContext&)
    /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/tools/arcilator/arcilator.cpp:629:14
#27 main
    /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/tools/arcilator/arcilator.cpp:697:8
```

## Root Cause Analysis

### Location
File: `lib/Dialect/Arc/Transforms/InferStateProperties.cpp:211`
Function: `applyEnableTransformation(circt::arc::DefineOp, circt::arc::StateOp, llvm::ArrayRef<EnableInfo>)`

### Problem
The `applyEnableTransformation` function attempts to create `hw::ConstantOp` instances to replace enable signal arguments:

```cpp
inputs[enableInfos[i].selfArg.getArgNumber()] = hw::ConstantOp::create(
    builder, stateOp.getLoc(), enableInfos[i].selfArg.getType(), 0);
```

However, `hw::ConstantOp::create` is only valid for integer types (`mlir::IntegerType` or `circt::hw::IntType`). When `enableInfos[i].selfArg.getType()` returns an array type (`hw::ArrayType`), the subsequent implicit cast to `mlir::IntegerType` in the return statement triggers the assertion failure.

### Why This Happens
When the InferStatePropertiesPass detects an "enable pattern" in state variables (which occurs with conditional assignments in `always_ff` blocks), it identifies arguments that need to be replaced with constant `0` values. However, it assumes all such arguments are integer-typed without validation.

In the testcase:
- Array-typed state variables (`packed_arr`, `unpacked_arr`) are identified as enable candidates
- Their types are `hw::ArrayType` (e.g., `!hw.array<2xi8>`)
- The code attempts to create a constant with this array type
- `hw::ConstantOp::create` rejects array types
- The cast assertion fails

### Proposed Fix

**Option 1: Type Guard (Recommended)**
Add a type check before calling `hw::ConstantOp::create`:

```cpp
auto argType = enableInfos[i].selfArg.getType();
if (!isa<IntegerType>(argType) && !isa<hw::IntType>(argType)) {
  // Skip or handle array/non-integer types appropriately
  // Option: Log a warning, or return failure()
  continue;
}
inputs[enableInfos[i].selfArg.getArgNumber()] = hw::ConstantOp::create(
    builder, stateOp.getLoc(), argType, 0);
```

**Option 2: Early Exit for Non-Integer Types**
Return failure from `applyEnableTransformation` if non-integer types are detected:

```cpp
auto argType = enableInfos[i].selfArg.getType();
if (!isa<IntegerType>(argType) && !isa<hw::IntType>(argType)) {
  return failure();
}
// ... rest of transformation
```

**Option 3: Pattern Filter (Most Comprehensive)**
Filter enable pattern detection to only consider integer-typed state variables:

```cpp
// In the enable pattern detection logic
if (!isa<IntegerType>(state.getType())) {
  // This state variable doesn't support enable patterns
  continue;
}
```

## Environment

- **CIRCT Version**: 1.139.0
- **LLVM Version**: Built with LLVM git (per build output)
- **Operating System**: Linux
- **Verification**:
  - slang: Pass ✅
  - verilator: Pass ✅
  - circt-verilog: Pass ✅
  - arcilator: Crash ❌

## Additional Context

- **Severity**: High (assertion failure, crash)
- **Classification**: Type safety / validation bug
- **Reproducibility**: 100% (consistently reproducible with testcase)
- **Duplicate Check**: No similar issues found in llvm/circt repository (searched for InferStateProperties, ConstantOp, IntegerType, ArrayType combinations)
- **Related Issues**: None with exact same crash pattern

## Minimization History

- **Original Testcase**: 749 bytes (36 lines)
- **Minimized Testcase**: 545 bytes (24 lines)
- **Reduction**: 27.2%

**Key findings during minimization**:
- Both packed and unpacked arrays are required to trigger the enable pattern detection
- For loop with array assignments is necessary
- Conditional assignment (if-else) is necessary to trigger enable pattern recognition
- Array size can be minimized to `[0:1]`
- Loop can be minimized to single iteration
