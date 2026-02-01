# Root Cause Analysis: CIRCT Arcilator Assertion Failure

## Summary

**Crash Type:** Assertion Failure  
**Affected Tool:** arcilator  
**Affected Dialect:** Arc  
**Root Cause:** LLHD reference type (`!llhd.ref<i1>`) passed to `arc::StateType::get()` which requires types with known bit width

## Error Message

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
```

## Stack Trace Analysis

The crash occurs in the Arc dialect's `LowerState` pass:

1. `LowerStatePass::runOnOperation()` (LowerState.cpp:1198)
2. `ModuleLowering::run()` (LowerState.cpp:219)
3. `StateType::get(mlir::Type)` (ArcTypes.cpp.inc:108)
4. `verifyInvariants` assertion fails (StorageUniquerSupport.h:180)

## Problematic Code in Test Case

```systemverilog
module MixPorts(
  input  logic        clk,
  input  logic        a,
  output logic        b,
  inout  wire         c    // <-- Bidirectional port (inout)
);
  // ...
  assign c = a ? 1'bz : 1'b0;  // <-- Tri-state assignment
endmodule
```

The `inout` wire `c` is a bidirectional port with tri-state logic (`1'bz`). When lowered through CIRCT's Verilog frontend (`circt-verilog --ir-hw`), this creates an LLHD reference type (`!llhd.ref<i1>`).

## Root Cause

### Location: `lib/Dialect/Arc/ArcTypes.cpp`

The `StateType::verify()` function validates that the inner type has a known bit width:

```cpp
LogicalResult StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                                Type innerType) {
  if (!computeLLVMBitWidth(innerType))
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

The `computeLLVMBitWidth()` function handles these types:
- `seq::ClockType`
- `IntegerType`
- `hw::ArrayType`
- `hw::StructType`

But it does **NOT** handle `llhd::RefType`. When an unknown type is encountered, it returns `{}` (nullopt), causing the verification to fail.

### Location: `lib/Dialect/Arc/Transforms/LowerState.cpp`

In `ModuleLowering::getAllocatedState()`:

```cpp
auto alloc = AllocStateOp::create(allocBuilder, result.getLoc(),
                                  StateType::get(result.getType()), // Line 219
                                  storageArg);
```

When `result.getType()` is `!llhd.ref<i1>` (from the inout port), `StateType::get()` triggers the verification which fails.

## Analysis

### Why This Happens

1. **Verilog Frontend:** `circt-verilog --ir-hw` converts `inout` ports to LLHD reference types to model bidirectional signals
2. **Arcilator Pipeline:** The LowerState pass attempts to allocate state storage for all values
3. **Type Incompatibility:** Arc's StateType was not designed to handle LLHD reference types

### The Actual Bug

This is a **missing validation/error handling** bug. The Arc dialect's `LowerState` pass should either:
1. **Emit a proper diagnostic** rejecting LLHD reference types before attempting to create StateType
2. **Handle LLHD reference types** appropriately (e.g., unwrap the reference to get the underlying type)
3. **Filter out inout signals** during the lowering process

The current code path leads to an assertion failure in the MLIR type system rather than a user-friendly error message.

## Impact

- **Severity:** Medium - Tool crash with assertion failure
- **User Impact:** Cannot simulate Verilog modules with `inout` ports using arcilator
- **Workaround:** Avoid `inout` ports when using arcilator

## Suggested Fixes

### Option 1: Add Validation in LowerState.cpp

Add early validation in `getAllocatedState()`:

```cpp
Value ModuleLowering::getAllocatedState(OpResult result) {
  // Check for unsupported types
  if (isa<llhd::RefType>(result.getType())) {
    result.getOwner()->emitError()
        << "arcilator does not support LLHD reference types (inout ports)";
    return {};
  }
  // ... existing code
}
```

### Option 2: Extend computeLLVMBitWidth()

Add support for LLHD reference types by unwrapping them:

```cpp
static std::optional<uint64_t> computeLLVMBitWidth(Type type) {
  // Handle LLHD reference types
  if (auto refType = dyn_cast<llhd::RefType>(type))
    return computeLLVMBitWidth(refType.getNestedType());
  // ... existing cases
}
```

## References

- `lib/Dialect/Arc/Transforms/LowerState.cpp:219` - Crash location
- `lib/Dialect/Arc/ArcTypes.cpp` - StateType verification
- `include/circt/Dialect/LLHD/IR/LLHDTypes.td` - LLHD RefType definition
