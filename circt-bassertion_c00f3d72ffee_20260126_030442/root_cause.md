# Root Cause Analysis Report

## Executive Summary

The arcilator tool crashes with an assertion failure when attempting to process a SystemVerilog module containing an `inout` port (bidirectional bus). The crash occurs in the `LowerState` pass when allocating storage for module inputs. The `StateType::get()` function fails because `llhd::RefType` (used to represent inout ports) is not a recognized type with known bit width in the Arc dialect's storage type system. This is a known limitation where inout ports are explicitly unsupported by the Arc dialect.

## Crash Context

- **Tool**: arcilator
- **Dialect**: Arc (with LLHD types from Moore conversion)
- **Failing Pass**: LowerStatePass
- **Crash Type**: Assertion failure (verifyInvariants)
- **CIRCT Version**: 1.139.0

## Error Analysis

### Assertion Message

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i8>'
arcilator: StorageUniquerSupport.h:180: Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### Key Stack Frames

```
#12 circt::arc::StateType::get(mlir::Type) /ArcTypes.cpp.inc:108
#13 (anonymous namespace)::ModuleLowering::run() /LowerState.cpp:219
#15 (anonymous namespace)::LowerStatePass::runOnOperation() /LowerState.cpp:1198
```

### Error Context

The crash is triggered at line 219 of `LowerState.cpp`:
```cpp
auto state = RootInputOp::create(allocBuilder, arg.getLoc(),
                                  StateType::get(arg.getType()), name, storageArg);
```

Where `arg.getType()` is `!llhd.ref<i8>` - representing the 8-bit inout port.

## Test Case Analysis

### Code Summary

The test case defines a module `MixedPorts` with a bidirectional 8-bit bus (`data_bus`) marked as `inout`. The module:
1. Has input clock (`clk`) and data (`data_in`)
2. Has output (`data_out`) 
3. Has bidirectional bus (`data_bus`) that conditionally drives or tristates based on `direction`
4. Contains an array with variable indexing
5. Uses `always_ff` for clocked write operations

### Key Constructs

- `inout logic [7:0] data_bus` - Bidirectional port (critical trigger)
- `logic [7:0] arr [0:15]` - Unpacked array
- Variable array indexing with `arr[idx]`
- Conditional tristate driver: `(direction) ? arr[idx] : 8'bz`
- Clocked array write in `always_ff`

### Problematic Patterns

The `inout` port declaration is the direct cause. When compiled through:
1. `circt-verilog --ir-hw` converts the inout port to a `hw::InOutType`
2. Moore-to-Core conversion wraps it as `llhd::RefType<i8>`
3. When arcilator's LowerState pass tries to allocate storage, it attempts `StateType::get(llhd::RefType<i8>)`
4. `StateType::verify()` calls `computeLLVMBitWidth()` which doesn't handle `llhd::RefType`, returning `nullopt`
5. Assertion failure

## CIRCT Source Analysis

### Crash Location

**File**: `lib/Dialect/Arc/Transforms/LowerState.cpp:219`
**Function**: `ModuleLowering::run()`

### Code Context

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

The loop iterates over all module arguments (inputs) and creates storage for each. For inout ports, `arg.getType()` is `llhd::RefType<innerType>` instead of a plain integer or array type.

### StateType Verification (ArcTypes.cpp:80-87)

```cpp
LogicalResult StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                                Type innerType) {
  if (!computeLLVMBitWidth(innerType))
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

### computeLLVMBitWidth (ArcTypes.cpp:29-76)

The function handles:
- `seq::ClockType` → 1 bit
- `IntegerType` → width
- `hw::ArrayType` → element width × count
- `hw::StructType` → sum of aligned element widths
- **All other types → `nullopt`** (including `llhd::RefType`)

### Arc ModelOp Verification (ArcOps.cpp:337-340)

```cpp
for (const hw::ModulePort &port : getIo().getPorts())
  if (port.dir == hw::ModulePort::Direction::InOut)
    return emitOpError("inout ports are not supported");
```

This verifier explicitly rejects inout ports, but the crash happens before model verification.

### Processing Path

1. **Parse**: `circt-verilog` parses SystemVerilog with inout port
2. **ImportVerilog**: Creates Moore dialect representation
3. **MooreToCore**: Converts inout to `llhd::RefType` (see FIXME comment at line 248-252)
4. **HW IR**: Module has argument with `llhd::RefType`
5. **arcilator LowerState**: Attempts `StateType::get(RefType)` → **CRASH**

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)

**Cause**: The Arc dialect's `StateType` cannot represent LLHD reference types (used for inout ports), but the pipeline allows such types to reach the LowerState pass without early rejection.

**Evidence**:
1. `computeLLVMBitWidth()` explicitly returns `nullopt` for all types except `ClockType`, `IntegerType`, `ArrayType`, and `StructType`
2. The error message clearly states: "state type must have a known bit width; got '!llhd.ref<i8>'"
3. Arc's `ModelOp::verify()` explicitly rejects inout ports, confirming this is a known unsupported feature
4. MooreToCore.cpp has FIXME comment (lines 248-252) noting that inout/ref ports are currently treated as inputs without proper handling

**Mechanism**:
```
inout port → MooreToCore → llhd::RefType → HW module argument → LowerState::run() 
→ StateType::get(RefType) → verify() → computeLLVMBitWidth(RefType) → nullopt 
→ assertion failure
```

### Hypothesis 2 (Medium Confidence)

**Cause**: Missing early validation pass that should reject modules with inout ports before reaching arcilator's LowerState pass.

**Evidence**:
1. Arc dialect explicitly states inout ports are unsupported in `ModelOp::verify()`
2. The crash happens during allocation, not during verification of the model
3. No pre-pass validation rejects the invalid input earlier in the pipeline

**Mechanism**: 
The verification that would catch this (`ModelOp::verify()`) runs after the `HWModuleOp` is transformed into a `ModelOp`, but the crash happens during that transformation.

### Hypothesis 3 (Low Confidence)

**Cause**: `computeLLVMBitWidth()` should be extended to handle `llhd::RefType` by extracting the nested type's width.

**Evidence**:
- `llhd::RefType` has a nested type that could have a known width
- The type `!llhd.ref<i8>` contains `i8` which has width 8

**Counter-evidence**:
- Inout ports semantically require bidirectional handling, not just storage allocation
- Simply extracting bit width would not solve the semantic issue of supporting bidirectional ports

## Suggested Fix Directions

1. **Preferred - Early Rejection**: Add a pre-validation pass before LowerState that emits a proper diagnostic when inout ports are detected:
   ```cpp
   if (isa<llhd::RefType>(arg.getType()))
     return moduleOp.emitOpError("inout ports are not supported by arcilator");
   ```

2. **Alternative - HW Pass**: Add a pass `arc-reject-inout-ports` that runs after HW lowering and before LowerState to catch this case with a user-friendly error message.

3. **Long-term - Feature Implementation**: Implement proper inout port support in the Arc dialect, which would require:
   - Bidirectional storage handling
   - Tristate logic support
   - LLVM backend changes for simulation

## Keywords for Issue Search

`arcilator`, `inout`, `llhd.ref`, `StateType`, `LowerState`, `computeLLVMBitWidth`, `bidirectional`, `tristate`

## Related Files

- `lib/Dialect/Arc/Transforms/LowerState.cpp` - Crash location
- `lib/Dialect/Arc/ArcTypes.cpp` - StateType verification and bit width computation
- `lib/Dialect/Arc/ArcOps.cpp` - ModelOp verification rejecting inout
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - FIXME for inout handling
- `include/circt/Dialect/Arc/ArcTypes.td` - StateType definition
- `include/circt/Dialect/LLHD/IR/LLHDTypes.td` - RefType definition
