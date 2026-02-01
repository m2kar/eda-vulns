# Root Cause Analysis Report

## Executive Summary

Arcilator crashes when processing a SystemVerilog module containing an `inout wire` port. The `inout` port is lowered to `!llhd.ref<i1>` type (LLHD reference type), which is unsupported by Arc dialect's `StateType`. The `StateType::get()` fails verification because `computeLLVMBitWidth()` returns `std::nullopt` for LLHD reference types.

## Crash Context

- **Tool/Command**: `circt-verilog --ir-hw ... | arcilator`
- **Dialect**: Arc (with LLHD ref type input)
- **Failing Pass**: `LowerStatePass` (arc-lower-state)
- **Crash Type**: Assertion failure in MLIR type uniquer

## Error Analysis

### Assertion/Error Message
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: .../mlir/include/mlir/IR/StorageUniquerSupport.h:180: 
Assertion `succeeded(ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### Key Stack Frames
```
#11 0x0000562d0cb68bbc  (StateType::get verification fails)
#12 0x0000562d0cb68ae9 circt::arc::StateType::get(mlir::Type) ArcTypes.cpp.inc:108
#13 0x0000562d0cbd3f5c ModuleLowering::run() LowerState.cpp:219
#14 0x0000562d0cbd3f5c LowerStatePass::runOnOperation() LowerState.cpp:1198
```

## Test Case Analysis

### Code Summary
```systemverilog
module MixPorts(
  input logic a,
  output logic b,
  inout wire c,        // <-- Problematic: bidirectional port
  input logic clk,
  output logic out0
);
  logic [3:0] idx;
  always_ff @(posedge clk) begin
    idx <= idx + 1;
  end
  assign out0 = idx[0];
  assign b = a;
  assign c = a ? 1'bz : 1'b0;  // <-- Tri-state driver
endmodule
```

### Key Constructs
- **`inout wire c`**: Bidirectional port that becomes `!llhd.ref<i1>` in IR
- **Tri-state assignment**: `c = a ? 1'bz : 1'b0` - high-impedance logic

### Potentially Problematic Patterns
The `inout` port is a feature that requires LLHD dialect's reference types to model bidirectional signal flow. However, the Arc dialect (arcilator) is designed for simulation of synthesizable RTL and does not support LLHD reference types.

## CIRCT Source Analysis

### Crash Location
**File**: `lib/Dialect/Arc/ArcTypes.cpp`  
**Function**: `StateType::verify()`  
**Lines**: 80-85

### Code Context
```cpp
// lib/Dialect/Arc/ArcTypes.cpp
LogicalResult
StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  Type innerType) {
  if (!computeLLVMBitWidth(innerType))  // <-- Returns nullopt for !llhd.ref
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

The `computeLLVMBitWidth()` function only handles:
- `seq::ClockType` → 1 bit
- `IntegerType` → width bits
- `hw::ArrayType` → recursive computation
- `hw::StructType` → recursive computation
- **All other types** → `std::nullopt` (unknown width)

### Processing Path
1. `circt-verilog --ir-hw` parses SystemVerilog and emits HW dialect IR
2. `inout wire c` is lowered to a port with `!llhd.ref<i1>` type
3. Arcilator runs `LowerStatePass` to convert HW modules to Arc models
4. At `LowerState.cpp:219`, `ModuleLowering::run()` allocates storage for inputs:
   ```cpp
   for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
     auto name = moduleOp.getArgName(arg.getArgNumber());
     auto state = RootInputOp::create(allocBuilder, arg.getLoc(),
                     StateType::get(arg.getType()),  // <-- CRASH HERE
                     name, storageArg);
   }
   ```
5. `StateType::get(!llhd.ref<i1>)` triggers type verification
6. `computeLLVMBitWidth(!llhd.ref<i1>)` returns `nullopt`
7. Verification fails → Assertion in StorageUniquerSupport.h:180

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: Arc dialect's `StateType` does not support LLHD reference types (`!llhd.ref<T>`), but the frontend pipeline can produce HW modules with such types for `inout` ports.

**Evidence**:
1. Error message explicitly states: `state type must have a known bit width; got '!llhd.ref<i1>'`
2. `computeLLVMBitWidth()` in ArcTypes.cpp has no case for `llhd::RefType`
3. The test case has `inout wire c` which requires bidirectional modeling

**Mechanism**: The `circt-verilog --ir-hw` lowering path produces `!llhd.ref<i1>` for inout ports. When arcilator attempts to allocate state storage for module inputs, it calls `StateType::get()` on all input types including the LLHD ref type, which fails verification.

### Hypothesis 2 (Medium Confidence)
**Cause**: Missing validation/rejection of unsupported port types before the LowerState pass.

**Evidence**:
1. Arcilator should either support LLHD refs or reject modules containing them early
2. No check exists in `LowerStatePass` to filter out unsupported port types
3. The error is emitted during type construction, not as a proper diagnostic

## Suggested Fix Directions

1. **Add early validation** in arcilator or `LowerStatePass` to reject modules with `!llhd.ref` typed ports with a clear error message explaining that inout ports are not supported.

2. **Extend `computeLLVMBitWidth()`** to handle `llhd::RefType` by extracting the inner type's width (if LLHD refs should be supported in simulation).

3. **Add a lowering pass** that converts LLHD ref types to a supported representation before the LowerState pass runs (e.g., splitting inout into separate input/output ports).

## Keywords for Issue Search
`arcilator` `StateType` `llhd.ref` `inout` `bit width` `LowerState` `computeLLVMBitWidth` `bidirectional` `tristate`

## Related Files to Investigate
- `lib/Dialect/Arc/ArcTypes.cpp` - StateType verification and computeLLVMBitWidth
- `lib/Dialect/Arc/Transforms/LowerState.cpp` - Module input lowering
- `tools/arcilator/arcilator.cpp` - Pipeline setup, potential validation point
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - How inout ports are lowered
