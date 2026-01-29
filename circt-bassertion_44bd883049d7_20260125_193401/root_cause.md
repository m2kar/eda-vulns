# Root Cause Analysis Report

## Executive Summary

**arcilator** crashes with an assertion failure when processing a SystemVerilog module containing `inout` ports. The `inout` port is lowered to LLHD `!llhd.ref<i1>` type, which cannot be converted to Arc `StateType` because `StateType` requires a type with known bit width. The `computeLLVMBitWidth()` function in ArcTypes.cpp does not handle reference types, causing the verification to fail.

## Crash Context

| Field | Value |
|-------|-------|
| Tool | arcilator |
| Crash Type | Assertion failure |
| Dialect | Arc (via HW/LLHD pipeline) |
| Failing Pass | LowerState (arc-lower-state) |
| Failing Operation | `StateType::get()` during `RootInputOp::create()` |

## Error Analysis

### Assertion Message
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### Key Stack Frames
```
#12 circt::arc::StateType::get(mlir::Type)         ArcTypes.cpp.inc:108
#13 (anonymous namespace)::ModuleLowering::run()   LowerState.cpp:219
#15 (anonymous namespace)::LowerStatePass::runOnOperation() LowerState.cpp:1198
```

## Test Case Analysis

### Code Summary
```systemverilog
module MixedPorts(
  input  logic clk,
  input  logic A,
  output logic O,
  inout  logic C  // <-- Problem: inout port
);
  always @(posedge clk) begin
    O = A;
  end
  assign C = A;
endmodule
```

### Key Constructs
- `inout` port declaration (`inout logic C`)
- Standard input/output ports
- Clocked always block with blocking assignment
- Continuous assignment to inout port

### Problematic Pattern
The `inout` (bidirectional) port `C` is the trigger. When lowered through the CIRCT pipeline:
1. `circt-verilog --ir-hw` converts `inout` to LLHD `!llhd.ref<T>` type
2. `arcilator` tries to allocate storage for module inputs via `RootInputOp`
3. `StateType::get(arg.getType())` is called with `!llhd.ref<i1>`
4. `StateType::verify()` fails because `computeLLVMBitWidth()` doesn't handle `llhd.ref`

## CIRCT Source Analysis

### Crash Location
**File:** `lib/Dialect/Arc/Transforms/LowerState.cpp:219`
```cpp
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(arg.getType()), name, storageArg);
  //                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^ CRASH HERE
  allocatedInputs.push_back(state);
}
```

### Type Verification Logic
**File:** `lib/Dialect/Arc/ArcTypes.cpp:81-86`
```cpp
LogicalResult
StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  Type innerType) {
  if (!computeLLVMBitWidth(innerType))   // Returns {} for llhd.ref
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

### Missing Type Handler
**File:** `lib/Dialect/Arc/ArcTypes.cpp:29-76`
```cpp
static std::optional<uint64_t> computeLLVMBitWidth(Type type) {
  if (isa<seq::ClockType>(type))
    return 1;
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  if (auto arrayType = dyn_cast<hw::ArrayType>(type))
    // ... handles arrays
  if (auto structType = dyn_cast<hw::StructType>(type))
    // ... handles structs
  
  // NO HANDLER for llhd::RefType or llhd::SigType
  return {};  // <-- Returns empty optional, causing verify failure
}
```

### Processing Path
1. `circt-verilog --ir-hw` parses SystemVerilog
2. `inout` port becomes `!llhd.ref<i1>` in HW module type
3. Pipeline to `arcilator`
4. `LowerStatePass::runOnOperation()` iterates HW modules
5. `ModuleLowering::run()` processes each module
6. For each block argument (port), creates `RootInputOp` with `StateType`
7. `StateType::get()` calls `verify()` which fails for ref types

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause:** Arc dialect's `StateType` does not support LLHD reference/signal types (`llhd.ref`, `llhd.sig`)

**Evidence:**
- Error message explicitly states: `state type must have a known bit width; got '!llhd.ref<i1>'`
- `computeLLVMBitWidth()` only handles: `seq::ClockType`, `IntegerType`, `hw::ArrayType`, `hw::StructType`
- No case for `llhd::RefType` or `llhd::SigType` exists

**Mechanism:**
The Arc dialect is designed for simulation and requires concrete bit widths for state storage allocation. LLHD reference types (`!llhd.ref<T>`) represent bidirectional signals that cannot be directly mapped to a fixed-width storage location. The LowerState pass assumes all module inputs can be stored as `StateType`, but `inout` ports violate this assumption.

### Hypothesis 2 (Medium Confidence)
**Cause:** Missing `inout` port filtering in the LowerState pass

**Evidence:**
- The loop at line 215-221 iterates over ALL block arguments without checking port direction
- `inout` ports should potentially be handled differently (or rejected early)

**Mechanism:**
The pass should either:
1. Reject modules with `inout` ports early with a clear error message
2. Handle `inout` ports specially (e.g., skip allocation or use different type)

### Hypothesis 3 (Low Confidence)
**Cause:** Incorrect type lowering in earlier passes

**Evidence:**
- The `--ir-hw` flag produces LLHD types for bidirectional ports
- There might be a missing pass or incorrect pass ordering

**Mechanism:**
Perhaps a pass should convert or strip LLHD types before arcilator's LowerState runs.

## Suggested Fix Directions

### Option 1: Add Early Rejection (Recommended)
Add a check in `ModuleLowering::run()` before the input allocation loop:
```cpp
// Before allocating inputs, check for unsupported types
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  if (isa<llhd::RefType, llhd::SigType>(arg.getType())) {
    return moduleOp.emitError()
        << "arcilator does not support inout/bidirectional ports; "
        << "port " << moduleOp.getArgName(arg.getArgNumber())
        << " has unsupported type " << arg.getType();
  }
}
```

### Option 2: Extend computeLLVMBitWidth
If LLHD ref types should be supported, add a handler:
```cpp
if (auto refType = dyn_cast<llhd::RefType>(type))
  return computeLLVMBitWidth(refType.getNestedType());
if (auto sigType = dyn_cast<llhd::SigType>(type))
  return computeLLVMBitWidth(sigType.getNestedType());
```
*Note: This may have semantic implications for simulation correctness.*

### Option 3: Filter inout Ports in LowerState
Skip `inout` ports during allocation:
```cpp
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  if (isa<llhd::RefType, llhd::SigType>(arg.getType()))
    continue;  // Skip bidirectional ports
  // ... existing allocation code
}
```
*Note: This may cause issues if the port is actually used in the design.*

## Keywords for Issue Search
`inout` `arcilator` `StateType` `llhd.ref` `LowerState` `bidirectional` `computeLLVMBitWidth`

## Related Files
- `lib/Dialect/Arc/Transforms/LowerState.cpp` - Crash location
- `lib/Dialect/Arc/ArcTypes.cpp` - Type verification logic
- `include/circt/Dialect/Arc/ArcTypes.td` - StateType definition
- `include/circt/Dialect/LLHD/IR/LLHDTypes.td` - RefType/SigType definitions
