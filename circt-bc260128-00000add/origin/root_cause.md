# Root Cause Analysis Report

## Executive Summary

Arcilator's `LowerStatePass` crashes when processing a SystemVerilog module with `inout` (bidirectional) port. The pass attempts to create an `arc::StateType` for an `!llhd.ref<i1>` type, but `StateType::verify()` fails because `!llhd.ref` is not a type with known bit width (not handled by `computeLLVMBitWidth()`). This indicates **arcilator does not support LLHD ref types** that arise from `inout` ports in the HW-to-Arc lowering path.

## Crash Context

- **Tool/Command**: `circt-verilog --ir-hw | arcilator`
- **Dialect**: Arc (with LLHD types present)
- **Failing Pass**: `arc::LowerStatePass`
- **Crash Type**: Assertion failure in type verification
- **CIRCT Version**: 1.139.0

## Error Analysis

### Assertion/Error Message
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### Key Stack Frames
```
#12 circt::arc::StateType::get(mlir::Type)
#13 (anonymous namespace)::ModuleLowering::run() LowerState.cpp:219
#14 (anonymous namespace)::LowerStatePass::runOnOperation() LowerState.cpp:1198
```

## Test Case Analysis

### Code Summary
```systemverilog
module test_module (inout logic io_sig);
  logic [1:0] out_val;
  
  always_comb begin
    out_val = 2'b01;
  end
  
  assign io_sig = (out_val[0]) ? 1'b1 : 1'bz;
endmodule
```

A simple module with:
1. **`inout` port** (`io_sig`) - bidirectional signal
2. **Tristate assignment** - `1'bz` high-impedance value

### Key Constructs
- `inout logic io_sig`: Bidirectional port, lowered to `!llhd.ref<i1>` in CIRCT
- `1'bz`: Tristate/high-impedance value

### Potentially Problematic Patterns
- **`inout` port**: LLHD dialect uses `llhd.ref` to represent bidirectional signals/references
- **Arcilator limitation**: Arc dialect's `StateType` only supports types with known bit widths (`IntegerType`, `seq::ClockType`, `hw::ArrayType`, `hw::StructType`)

## CIRCT Source Analysis

### Crash Location
**File**: `lib/Dialect/Arc/ArcTypes.cpp`
**Function**: `StateType::verify()`
**Line**: ~78-82

### Code Context
```cpp
// lib/Dialect/Arc/ArcTypes.cpp
static std::optional<uint64_t> computeLLVMBitWidth(Type type) {
  if (isa<seq::ClockType>(type))
    return 1;

  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();

  if (auto arrayType = dyn_cast<hw::ArrayType>(type)) { ... }
  if (auto structType = dyn_cast<hw::StructType>(type)) { ... }

  // We don't know anything about any other types.
  return {};  // <-- Returns nullopt for !llhd.ref<i1>
}

LogicalResult
StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  Type innerType) {
  if (!computeLLVMBitWidth(innerType))
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

### Processing Path
1. `circt-verilog --ir-hw` parses SystemVerilog and generates HW/LLHD IR
2. `inout` port is represented as `!llhd.ref<i1>` in the IR
3. Arcilator runs `LowerStatePass` to convert HW modules to Arc models
4. `ModuleLowering::run()` (line 219) allocates storage for module inputs:
   ```cpp
   for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
     auto state = RootInputOp::create(allocBuilder, arg.getLoc(),
                                      StateType::get(arg.getType()), ...);
   }
   ```
5. `StateType::get(arg.getType())` is called with `!llhd.ref<i1>` type
6. `StateType::verify()` fails because `computeLLVMBitWidth(!llhd.ref<i1>)` returns `nullopt`

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: Arcilator does not support bidirectional (`inout`) ports represented as `!llhd.ref` types.

**Evidence**:
- Error message explicitly states: `state type must have a known bit width; got '!llhd.ref<i1>'`
- `computeLLVMBitWidth()` only handles `seq::ClockType`, `IntegerType`, `hw::ArrayType`, `hw::StructType`
- No case for `llhd::RefType` exists in the function
- `inout` ports in SystemVerilog are lowered to `!llhd.ref` by `circt-verilog`

**Mechanism**: 
When `LowerStatePass` tries to allocate state storage for module inputs, it creates `StateType::get(arg.getType())`. For `inout` ports, `arg.getType()` is `!llhd.ref<i1>`, which fails the `StateType` verification invariant.

### Hypothesis 2 (Medium Confidence)
**Cause**: The HW-to-Arc lowering path should reject or transform `!llhd.ref` types before reaching `StateType::get()`.

**Evidence**:
- Arc dialect is designed for simulation, where bidirectional ports have different semantics
- The error occurs deep in type creation rather than at an earlier validation point
- No graceful error handling exists for unsupported port types

**Mechanism**:
Missing input validation in `LowerStatePass` allows unsupported port types to reach internal type construction, causing assertion failures instead of user-friendly errors.

## Suggested Fix Directions

1. **Add `llhd::RefType` support to `computeLLVMBitWidth()`** (if simulation semantics allow):
   ```cpp
   if (auto refType = dyn_cast<llhd::RefType>(type))
     return computeLLVMBitWidth(refType.getNestedType());
   ```

2. **Add early validation in `LowerStatePass`** to reject modules with `inout` ports:
   ```cpp
   for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
     if (isa<llhd::RefType>(arg.getType())) {
       return moduleOp.emitError() 
         << "inout ports are not supported by arcilator";
     }
   }
   ```

3. **Add documentation** that arcilator does not support bidirectional ports.

## Keywords for Issue Search
`arcilator` `inout` `llhd.ref` `StateType` `LowerStatePass` `bit width` `bidirectional` `tristate`

## Related Files to Investigate
- `lib/Dialect/Arc/ArcTypes.cpp` - `computeLLVMBitWidth()` function
- `lib/Dialect/Arc/Transforms/LowerState.cpp` - Pass implementation, line 219
- `include/circt/Dialect/Arc/ArcTypes.td` - `StateType` definition with `genVerifyDecl = 1`
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - How `inout` ports are lowered

## Classification

| Aspect | Value |
|--------|-------|
| Bug Type | Incomplete feature support |
| Severity | Medium - triggers assertion, clear workaround (avoid `inout`) |
| User Impact | Cannot simulate modules with bidirectional ports using arcilator |
| Reproducibility | Deterministic |
