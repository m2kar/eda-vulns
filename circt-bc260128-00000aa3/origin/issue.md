# [arcilator] Assertion failure in InferStateProperties pass with unpacked arrays

## Bug Description

The `arcilator` tool crashes with an assertion failure when processing SystemVerilog code that contains unpacked arrays with shift register patterns. The crash occurs in the `InferStateProperties` pass at line 211 of `lib/Dialect/Arc/Transforms/InferStateProperties.cpp` when the `applyEnableTransformation` function attempts to create an `hw::ConstantOp` with a non-integer array type.

### Error Message

```
arc.state op operand type mismatch: operand #2
<stdin>:36:10: error: 'arc.state' op operand type mismatch: operand #2
    %2 = comb.mux bin %0#1, %0#0, %a : !hw.array<2xi1>
         ^
<stdin>:36:10: note: see current operation: %3 = "arc.state"(%7, %0#1, %1, %0#0, %2) <{arc = @m_arc, latency = 1 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 3, 0>}> : (!seq.clock, i1, i1, !hw.array<2xi1>, i1062869376) -> !hw.array<2xi1>
<stdin>:36:10: note: expected type: '!hw.array<2xi1>'
<stdin>:36:10: note:   actual type: 'i1062869376' (corrupted/garbage type)
```

The original crash from the older version shows the assertion failure:

```
arcilator: /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/Casting.h:566: decltype(auto) llvm::cast(From &) [To = mlir::IntegerType, From = mlir::Type]: Assertion `isa<To>(Val) && "cast<Ty>() argument of incompatible type!"' failed.
```

## Minimized Test Case

```systemverilog
module m(input clk, output logic out);
  logic a [0:1];
  always_ff @(posedge clk) begin
    out <= a[0];
    a[0] <= 0;
    for (int i = 1; i < 2; i++)
      a[i] <= a[i-1];
  end
endmodule
```

### Reproduction Command

```bash
export PATH=/opt/llvm-22/bin:/opt/firtool-1.139.0/bin:$PATH
/opt/firtool-1.139.0/bin/circt-verilog --ir-hw bug.sv 2>&1 | \
/opt/firtool-1.139.0/bin/arcilator 2>&1
```

## Root Cause Analysis

The crash occurs in the `InferStateProperties` pass's `applyEnableTransformation` function at line 211:

```cpp
for (size_t i = 0, e = outputOp.getOutputs().size(); i < e; ++i) {
  if (enableInfos[i].selfArg.hasOneUse())
    inputs[enableInfos[i].selfArg.getArgNumber()] = hw::ConstantOp::create(
        builder, stateOp.getLoc(), enableInfos[i].selfArg.getType(), 0);  // CRASH HERE
}
```

### The Problem

The code assumes that all `selfArg` types are scalar integers when creating a zero constant via `hw::ConstantOp::create()`. However, when processing unpacked arrays with shift register patterns:

1. The array `logic a [0:1]` creates state operations with **array type** outputs (`!hw.array<2xi1>`)
2. The shift register pattern (`a[i] <= a[i-1]`) triggers enable pattern detection
3. `enableInfos[i].selfArg.getType()` returns an **array type**, not an `IntegerType`
4. `hw::ConstantOp::create()` internally calls `cast<mlir::IntegerType>(type)` which fails with an assertion

### Why This Happens

The `hw::ConstantOp` only accepts `mlir::IntegerType` or `circt::hw::IntType`, but the `InferStateProperties` pass does not check the type before attempting to create a constant. When unpacked arrays are lowered through the Arc dialect, they produce aggregate types that violate this type constraint.

## Affected Components

- **Tool**: `arcilator`
- **Pass**: `InferStateProperties` (Arc dialect transformation)
- **File**: `lib/Dialect/Arc/Transforms/InferStateProperties.cpp:211`
- **Function**: `applyEnableTransformation`

## Suggested Fix

Add a type check before creating the constant:

```cpp
for (size_t i = 0, e = outputOp.getOutputs().size(); i < e; ++i) {
  if (enableInfos[i].selfArg.hasOneUse()) {
    Type argType = enableInfos[i].selfArg.getType();
    // Only create constant for integer types
    if (isa<mlir::IntegerType>(argType) || isa<hw::IntType>(argType)) {
      inputs[enableInfos[i].selfArg.getArgNumber()] = hw::ConstantOp::create(
          builder, stateOp.getLoc(), argType, 0);
    } else {
      // Cannot optimize - bail out for non-integer types (arrays, structs, etc.)
      return failure();
    }
  }
}
```

## Additional Information

### Test Case Validation

The minimized test case is valid SystemVerilog and is accepted by:
- ✅ `slang` (verilog frontend)
- ✅ `verilator` (lint check)
- ✅ `circt-verilog` (parsing and HW IR generation)

The crash specifically occurs in the `arcilator` lowering stage.

### Related Issues

This bug is **not a duplicate** of existing issues, although it shares some similarities:
- #9469: Different root cause (llhd.constant_time legalization vs. type casting)
- #9395: Arcilator assertion but different error location
- #9417: Aggregate type handling but in hw.bitcast, not InferStateProperties
- #7627: Unpacked array crash in Moore dialect (different component)

The unique aspect is the type casting assertion failure in `applyEnableTransformation` when handling unpacked arrays with shift register patterns.

### Toolchain Version

```
circt-verilog: firtool-1.139.0
arcilator: firtool-1.139.0
LLVM: 22.0.0git
```
