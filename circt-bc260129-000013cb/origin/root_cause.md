# Root Cause Analysis: Arc LowerState Crash on `llhd.ref<i1>` Type

## Summary

The `arcilator` tool crashes with an assertion failure when processing a SystemVerilog module containing an **inout port** with a conditional tristate assignment. The crash occurs in the `LowerState` pass when attempting to create a `StateType` from an `llhd.ref<i1>` type, which is not a supported type for Arc state allocation.

## Crash Details

- **Error Message**: `state type must have a known bit width; got '!llhd.ref<i1>'`
- **Crash Location**: `lib/Dialect/Arc/Transforms/LowerState.cpp:219`
- **Function**: `ModuleLowering::run()`
- **Tool**: `arcilator`

## Test Case Analysis

### Source Code (`source.sv`)
```systemverilog
module TopModule (
  input wire clk,
  input logic [63:0] wide_input,
  output reg [7:0] out,
  inout logic io_sig
);

  always @(posedge clk) begin
    out <= wide_input[7:0] ^ wide_input[15:8];
  end

  assign io_sig = (wide_input[0]) ? out[0] : 1'bz;

endmodule
```

### Key Observations

1. **Inout Port**: The module has an `inout logic io_sig` port
2. **Tristate Logic**: The port is driven with conditional Z-state: `(wide_input[0]) ? out[0] : 1'bz`
3. **Signal Reference**: In the CIRCT IR lowering pipeline, inout ports are represented as `llhd.ref<T>` types to model bidirectional signals

## Code Location in CIRCT

### Crash Point (`lib/Dialect/Arc/Transforms/LowerState.cpp:215-221`)
```cpp
// Allocate storage for the inputs.
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(arg.getType()), name, storageArg);
  allocatedInputs.push_back(state);
}
```

The code iterates over all module arguments (ports) and attempts to allocate state storage for each. It calls `StateType::get(arg.getType())` without checking if the type is compatible with `StateType`.

### StateType Verification (`lib/Dialect/Arc/ArcTypes.cpp:80-87`)
```cpp
LogicalResult
StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  Type innerType) {
  if (!computeLLVMBitWidth(innerType))
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

The `StateType::verify()` function uses `computeLLVMBitWidth()` to check if the inner type has a known bit width.

### Supported Types in `computeLLVMBitWidth()` (`lib/Dialect/Arc/ArcTypes.cpp:29-76`)

The function only supports:
- `seq::ClockType` → 1 bit
- `IntegerType` → width from type
- `hw::ArrayType` → recursive computation
- `hw::StructType` → recursive computation

**Notably absent**: `llhd::RefType` is not handled, so it returns `std::nullopt`, causing the verification to fail.

## Root Cause

The **`LowerState` pass does not handle `llhd.ref<T>` types** when allocating storage for module inputs/outputs. When a SystemVerilog module has an `inout` port, the CIRCT frontend (likely via `circt-verilog --ir-hw`) represents it as an `llhd.ref<T>` type to model the bidirectional signal semantics.

The `StateType::get()` function has an invariant that the inner type must have a known bit width computable by `computeLLVMBitWidth()`. Since `llhd.ref<T>` is not in the list of supported types, the verification fails and triggers the assertion.

### Why This Is a Bug

1. **Missing Type Check**: The `LowerState` pass should either:
   - Filter out `llhd.ref<T>` types before attempting to allocate state
   - Handle `llhd.ref<T>` types appropriately (e.g., by extracting the nested type)
   - Emit a proper diagnostic rather than crashing

2. **Incomplete Type Coverage**: The `computeLLVMBitWidth()` function should either handle `llhd.ref<T>` or the code that calls `StateType::get()` should validate the type first.

## Suggested Fix Directions

### Option 1: Filter Unsupported Types in LowerState

Before calling `StateType::get()`, check if the type is supported:
```cpp
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto type = arg.getType();
  // Skip llhd.ref types (inout ports) - not supported in Arc simulation
  if (isa<llhd::RefType>(type)) {
    emitError(arg.getLoc()) << "inout ports are not supported by arcilator";
    return failure();
  }
  // ... existing code
}
```

### Option 2: Extend `computeLLVMBitWidth()` to Handle RefType

```cpp
if (auto refType = dyn_cast<llhd::RefType>(type))
  return computeLLVMBitWidth(refType.getNestedType());
```

### Option 3: Handle RefType Specially in LowerState

Unwrap the reference type and allocate state for the underlying type, while maintaining reference semantics during simulation.

## Conclusion

This is an **invariant violation bug** in the Arc dialect's `LowerState` pass. The pass assumes all module port types can be directly converted to `StateType`, but this assumption fails for `llhd.ref<T>` types that represent inout ports from SystemVerilog. The fix should either explicitly reject unsupported types with a clear diagnostic or extend the type handling to support `llhd.ref<T>`.
