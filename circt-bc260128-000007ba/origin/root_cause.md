# Root Cause Analysis Report

## Executive Summary

Arcilator crashes with an assertion failure when processing a module containing an `inout` port. The `!llhd.ref<i1>` type used to represent bidirectional ports is not supported by `arc::StateType`, which requires types with a known bit width that `computeLLVMBitWidth()` can calculate.

## Crash Context

- **Tool/Command**: `arcilator` (pipeline: `circt-verilog --ir-hw` | `arcilator`)
- **Dialect**: Arc (target), LLHD (source of problematic type)
- **Failing Pass**: `LowerStatePass` in Arc dialect
- **Crash Type**: Assertion failure in type verification

## Error Analysis

### Assertion/Error Message
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### Key Stack Frames
```
#12 circt::arc::StateType::get(mlir::Type) ArcTypes.cpp.inc:108
#13 (anonymous namespace)::ModuleLowering::run() LowerState.cpp:219
#14 (anonymous namespace)::LowerStatePass::runOnOperation() LowerState.cpp:1198
```

## Test Case Analysis

### Code Summary
```systemverilog
module MixPorts(
  input logic clk,
  input logic [63:0] wide_input,
  output logic [31:0] out_val,
  inout logic io_sig            // <-- Problematic port
);
  // ... combinational and sequential logic
endmodule
```

### Key Constructs
- **`inout logic io_sig`**: Bidirectional port that gets converted to `!llhd.ref<i1>` type

### Potentially Problematic Patterns
- Mixing unidirectional (`input`/`output`) with bidirectional (`inout`) ports
- The `inout` port type is converted to LLHD reference type which arcilator cannot handle

## CIRCT Source Analysis

### Crash Location
**File**: `lib/Dialect/Arc/Transforms/LowerState.cpp`
**Function**: `ModuleLowering::run()`
**Line**: 219

### Code Context
```cpp
// LowerState.cpp:214-221
// Allocate storage for the inputs.
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(arg.getType()), name, storageArg);  // Line 219
  allocatedInputs.push_back(state);
}
```

The code iterates over all module arguments (ports) and creates `RootInputOp` with `StateType` wrapping the argument's type.

### Type Verification
```cpp
// ArcTypes.cpp:80-87
LogicalResult
StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  Type innerType) {
  if (!computeLLVMBitWidth(innerType))
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

### computeLLVMBitWidth Limitations
```cpp
// ArcTypes.cpp:29-76
static std::optional<uint64_t> computeLLVMBitWidth(Type type) {
  // Supports: seq::ClockType, IntegerType, hw::ArrayType, hw::StructType
  // Does NOT support: llhd::RefType, llhd::SigType, and other types
  return {};  // Returns nullopt for unsupported types
}
```

### Processing Path
1. SystemVerilog `inout` port → LLHD `!llhd.ref<T>` type in IR
2. `LowerStatePass` processes module inputs
3. For each input, creates `RootInputOp` with `StateType::get(arg.getType())`
4. `StateType::get()` calls `verify()` → `computeLLVMBitWidth()`
5. `!llhd.ref<i1>` returns `std::nullopt` (unsupported type)
6. Verification fails → Assertion triggered

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: Arcilator's `StateType` does not support LLHD reference types (`!llhd.ref<T>`)

**Evidence**:
- Error message explicitly states: `state type must have a known bit width; got '!llhd.ref<i1>'`
- `computeLLVMBitWidth()` only handles `ClockType`, `IntegerType`, `ArrayType`, `StructType`
- `llhd::RefType` is not in the supported type list

**Mechanism**: 
When a SystemVerilog `inout` port is lowered, it becomes an `!llhd.ref<T>` type. The LowerState pass attempts to wrap all module arguments in `StateType`, but `StateType::verify()` rejects types that `computeLLVMBitWidth()` cannot compute, causing the assertion.

### Hypothesis 2 (Medium Confidence)
**Cause**: Missing pre-pass filtering or conversion for bidirectional ports

**Evidence**:
- The loop at line 215 processes ALL block arguments without filtering
- No check for whether the port type is supported by arcilator
- Bidirectional ports may need different handling or rejection earlier

**Mechanism**:
The LowerState pass assumes all input ports can be converted to Arc state storage. Bidirectional ports represented as LLHD references should either be converted to a different representation or rejected with a clear diagnostic before reaching this point.

## Suggested Fix Directions

1. **Add support for `llhd::RefType` in `computeLLVMBitWidth()`**: Compute the bit width of the inner type for reference types
   
2. **Add early validation in LowerState**: Check port types before attempting to create `StateType` and emit a proper diagnostic for unsupported types

3. **Filter out inout ports in earlier passes**: If arcilator cannot simulate bidirectional ports, they should be rejected or stripped earlier in the pipeline with a clear error message

4. **Document limitation**: If inout ports are intentionally unsupported, document this limitation clearly

## Keywords for Issue Search
`arcilator` `LowerState` `StateType` `inout` `llhd.ref` `bit width` `verifyInvariants` `computeLLVMBitWidth`

## Related Files to Investigate
- `lib/Dialect/Arc/Transforms/LowerState.cpp` - The crashing pass
- `lib/Dialect/Arc/ArcTypes.cpp` - StateType verification logic
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Where inout becomes llhd.ref
- `include/circt/Dialect/LLHD/IR/LLHDTypes.td` - LLHD type definitions
