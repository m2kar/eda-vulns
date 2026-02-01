# [Arc][arcilator] Assertion failure in InferStateProperties pass when using packed struct types

## Bug Description

The `arc-infer-state-properties` pass in `arcilator` crashes with an assertion failure when processing SystemVerilog code that uses packed struct types in state elements. The crash occurs at `InferStateProperties.cpp:211` in the `applyEnableTransformation()` function when it attempts to create a `hw::ConstantOp` with a packed struct type, but `hw::ConstantOp` only supports integer types (`IntegerType` or `hw::IntType`).

## Testcase

```systemverilog
module bug(input logic clk, input logic [7:0] in);
  typedef struct packed {
    logic [7:0] a;
  } pkt_t;

  pkt_t arr [0:1];
  logic [7:0] d;

  always_ff @(posedge clk) begin
    d <= in;
    arr[0].a <= d;
    for (int i = 1; i < 2; i++)
      arr[i].a <= arr[i-1].a;
  end
endmodule
```

## Steps to Reproduce

1. Save the testcase above as `bug.sv`
2. Run: `circt-verilog --ir-hw bug.sv | arcilator`
3. Observe the assertion failure

## Error Output

```
arcilator: /path/to/circt-src/llvm/llvm/include/llvm/Support/Casting.h:566: decltype(auto) llvm::cast(From &) [To = mlir::IntegerType, From = mlir::Type]: Assertion `isa<To>(Val) && "cast<Ty>() argument of incompatible type!"' failed.

Stack trace:
#12 circt::hw::ConstantOp::create(mlir::OpBuilder&, mlir::Location, mlir::Type, long)
#17 applyEnableTransformation(...) at InferStateProperties.cpp:211:55
#18 InferStatePropertiesPass::runOnStateOp(...) at InferStateProperties.cpp:454:17
```

## Root Cause Analysis

The `InferStateProperties` pass detects an "enable pattern" (mux selecting between new value and previous state) and attempts to replace the self-argument with a constant zero. At line 211 of `InferStateProperties.cpp`:

```cpp
inputs[enableInfos[i].selfArg.getArgNumber()] = hw::ConstantOp::create(
    builder, stateOp.getLoc(), enableInfos[i].selfArg.getType(), 0);
```

The code passes `enableInfos[i].selfArg.getType()` as the type parameter. When the state element is a packed struct (e.g., `!hw.struct<a: i8>`), this type is **not** an `IntegerType`. The `hw::ConstantOp` operation is constrained to only produce integer types:

```cpp
mlir::OpTrait::OneTypedResult<circt::hw::TypeVariant<mlir::IntegerType, circt::hw::IntType>>
```

This causes the internal `cast<IntegerType>()` to fail with an assertion.

## Trigger Conditions

The crash requires ALL of the following:
1. **Packed struct** type used in unpacked array as a state element
2. **Intermediate register** to create an enable pattern
3. **For-loop** that generates mux-based enable detection
4. The loop updates struct array elements conditionally

## Affected Components

- **Tool**: arcilator
- **Pass**: arc-infer-state-properties
- **File**: lib/Dialect/Arc/Transforms/InferStateProperties.cpp
- **Function**: applyEnableTransformation()

## Suggested Fix

Add a type check in `applyEnableTransformation()` before attempting to create `hw::ConstantOp`:

```cpp
// Check if type is compatible with hw::ConstantOp
if (!isa<IntegerType>(enableInfos[i].selfArg.getType()) &&
    !isa<hw::IntType>(enableInfos[i].selfArg.getType())) {
  // Skip transformation for non-integer types (e.g., struct, array)
  return failure();
}
```

Alternatively, use `hw::AggregateConstantOp` for struct types or implement struct-aware constant operations.

## Related Issues

- #6373 - "[Arc] Support hw.wires of aggregate types" - Related issue involving struct type handling in arc dialect, but focuses on `arc.tap` operations rather than `hw::ConstantOp` creation.

## Additional Context

- **Reproducibility**: Deterministic (crashes on every run)
- **Severity**: High (deterministic crash on valid SystemVerilog code)
- **Valid Code**: Testcase passes syntax validation in both verilator and slang
- **Features Used**: packed struct, unpacked array, always_ff, for-loop (all supported by CIRCT)

## Version Information

- CIRCT version: 1.139.0 (and current development version)
- Testcase ID: 260128-0000114b
