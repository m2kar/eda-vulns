# Root Cause Analysis Report

## Executive Summary
The arcilator tool crashes when processing SystemVerilog code containing an `inout` (bidirectional/tristate) port. The LLHD dialect represents the `inout` port as `llhd.ref<i1>` type, but when the Arc dialect's LowerState pass attempts to allocate state storage for this type, it fails because `StateType` requires a type with known bit width, and LLHD reference types are not handled by the bit-width computation function.

## Crash Context
- **Tool**: arcilator (via pipeline: circt-verilog → arcilator → opt → llc)
- **Dialect**: Arc (arcilator), with LLHD types from Moore dialect lowering
- **Failing Pass**: LowerStatePass (arc-lower-state)
- **Crash Type**: Assertion failure
- **CIRCT Version**: 1.139.0

## Error Analysis

### Error Message
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
```

### Assertion Message
```
Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

This assertion is triggered in `StateType::get()` when the verification of invariants fails.

### Key Stack Frames
```
#12 circt::arc::StateType::get(mlir::Type) ArcTypes.cpp.inc:108
#13 (anonymous namespace)::ModuleLowering::run() LowerState.cpp:219
#15 (anonymous namespace)::LowerStatePass::runOnOperation() LowerState.cpp:1198
```

## Test Case Analysis

### Code Summary
```systemverilog
module MixedPorts(
  input logic a,
  output logic b,
  inout wire c,                    // <- Problematic: bidirectional port
  input logic [`WIDTH-1:0] data_in
);
  always_comb begin
    b = (data_in == 0);
  end
  assign c = a ? data_in[0] : 1'bz;  // <- Tristate driver
endmodule
```

The module demonstrates:
1. Mixed port directions (input, output, inout)
2. Tristate driver using high-impedance value (`1'bz`)
3. Macro-based width definition

### Key Constructs
- `inout wire c` - Bidirectional port declaration
- `1'bz` - High-impedance literal for tristate logic
- Conditional assignment for tristate control

### Problematic Patterns
The `inout` port `c` is the root cause. In the CIRCT flow:
1. `circt-verilog` parses the SystemVerilog and produces Moore dialect IR
2. Moore dialect converts `inout` ports to LLHD reference types (`!llhd.ref<T>`)
3. When piped to `arcilator`, the Arc dialect's LowerState pass tries to allocate storage
4. `StateType::get()` is called with `!llhd.ref<i1>` type
5. The verification fails because LLHD ref types have no known bit width

## CIRCT Source Analysis

### Crash Location
**File**: `lib/Dialect/Arc/ArcTypes.cpp`  
**Function**: `StateType::verify()`

### Root Cause in Code
```cpp
// lib/Dialect/Arc/ArcTypes.cpp
static std::optional<uint64_t> computeLLVMBitWidth(Type type) {
  if (isa<seq::ClockType>(type))
    return 1;
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  if (auto arrayType = dyn_cast<hw::ArrayType>(type))
    // ... handles arrays
  if (auto structType = dyn_cast<hw::StructType>(type))
    // ... handles structs
  // We don't know anything about any other types.
  return {};  // <- LLHD ref types fall through here!
}

LogicalResult StateType::verify(..., Type innerType) {
  if (!computeLLVMBitWidth(innerType))
    return emitError() << "state type must have a known bit width; got " 
                       << innerType;
  return success();
}
```

The `computeLLVMBitWidth()` function only handles:
- `seq::ClockType`
- `IntegerType`
- `hw::ArrayType`
- `hw::StructType`

It does NOT handle `llhd::RefType`, causing the verification to fail.

### Processing Path
1. `circt-verilog` parses SystemVerilog with `--ir-hw` flag
2. For `inout` ports, the Moore dialect creates LLHD reference types
3. `arcilator` runs `LowerStatePass` on the module
4. `ModuleLowering::run()` line 219 tries to allocate storage for inputs:
   ```cpp
   auto state = RootInputOp::create(allocBuilder, arg.getLoc(),
                                    StateType::get(arg.getType()), // <- fails here
                                    name, storageArg);
   ```
5. `StateType::get(arg.getType())` triggers verification with `!llhd.ref<i1>`
6. Verification fails, assertion triggers

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: Arc dialect's `StateType` does not support LLHD reference types, but `circt-verilog --ir-hw` produces LLHD ref types for `inout` ports.

**Evidence**:
- Error message explicitly states: `state type must have a known bit width; got '!llhd.ref<i1>'`
- `computeLLVMBitWidth()` has no handler for LLHD types
- Test case contains `inout wire c` which maps to LLHD ref type
- The crash occurs in `ModuleLowering::run()` when allocating input storage

**Confidence**: High (95%)

### Hypothesis 2 (Medium Confidence)
**Cause**: The `--ir-hw` flag in `circt-verilog` is supposed to produce HW dialect output, but `inout` ports still leak through as LLHD types, indicating incomplete conversion.

**Evidence**:
- The flag name suggests HW dialect output
- LLHD types appearing in the output suggest incomplete lowering
- Related GitHub issue mentions users getting "unexpected dialects including... llhd"

**Confidence**: Medium (70%)

### Hypothesis 3 (Low Confidence)
**Cause**: Tristate/bidirectional logic is fundamentally not supported in the arcilator simulation flow.

**Evidence**:
- arcilator is designed for cycle-based simulation
- Tristate logic requires special handling (X/Z states)
- The error message is a diagnostic rather than a user-friendly "not supported" message

**Confidence**: Low (40%)

## Suggested Fix Directions

1. **Add LLHD Type Handling in Arc** (Short-term workaround):
   - Extend `computeLLVMBitWidth()` to handle `llhd::RefType` by extracting the underlying type
   - However, this doesn't address the semantic mismatch of reference vs value types

2. **Proper Conversion in Moore-to-HW** (Correct fix):
   - Ensure `--ir-hw` flag properly converts all LLHD types to HW equivalents
   - For `inout` ports, either reject them with a clear error or convert to appropriate HW representation

3. **Better Error Message** (User experience):
   - Detect unsupported types early in arcilator pipeline
   - Emit clear diagnostic: "arcilator does not support bidirectional (inout) ports"

4. **Documentation** (Minimum):
   - Document that arcilator doesn't support tristate/inout ports
   - Provide guidance on alternative simulation approaches

## Keywords for Issue Search
`StateType` `arcilator` `inout` `llhd.ref` `bidirectional` `tristate` `LowerState` `bit width` `computeLLVMBitWidth`

## Related Files
- `lib/Dialect/Arc/ArcTypes.cpp` - StateType verification
- `lib/Dialect/Arc/Transforms/LowerState.cpp` - LowerState pass
- `lib/Conversion/MooreToCore/` - Moore to HW/LLHD conversion
- `include/circt/Dialect/Arc/ArcTypes.td` - StateType definition
