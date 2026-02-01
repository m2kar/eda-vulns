# Root Cause Analysis: CIRCT Arcilator Crash

## Crash Summary

| Field | Value |
|-------|-------|
| **Crash ID** | 260129-000017c2 |
| **Crash Type** | Assertion Failure |
| **Tool** | arcilator |
| **Dialect** | Arc |
| **Component** | LowerState pass |
| **Crash Location** | `LowerState.cpp:219` → `StateType::get()` |

## Error Message

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
```

## Stack Trace Analysis

```
#12 circt::arc::StateType::get(mlir::Type) 
    /target/circt-1.139.0-src/build/tools/circt/include/circt/Dialect/Arc/ArcTypes.cpp.inc:108
#13 (anonymous namespace)::ModuleLowering::run() 
    /target/circt-1.139.0-src/lib/Dialect/Arc/Transforms/LowerState.cpp:219
#14 (anonymous namespace)::LowerStatePass::runOnOperation() 
    /target/circt-1.139.0-src/lib/Dialect/Arc/Transforms/LowerState.cpp:1198
```

The crash occurs in `ModuleLowering::run()` at line 219 when attempting to create a `StateType` for module inputs. The specific line is:

```cpp
auto state =
    RootInputOp::create(allocBuilder, arg.getLoc(),
                        StateType::get(arg.getType()), name, storageArg);
```

## Root Cause Identification

### Direct Cause

The `StateType::verify()` function in `ArcTypes.cpp` rejects any type that doesn't have a computable LLVM bit width:

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

The `computeLLVMBitWidth()` function only handles:
- `seq::ClockType`
- `IntegerType`  
- `hw::ArrayType`
- `hw::StructType`

The type `!llhd.ref<i1>` (LLHD Reference Type) is **not** among the supported types, causing `computeLLVMBitWidth()` to return an empty optional (`{}`), which triggers the verification failure.

### Upstream Cause

When SystemVerilog code with `inout` ports is processed:

1. `circt-verilog --ir-hw` converts the SystemVerilog to CIRCT's HW dialect
2. `inout` ports are represented using `!llhd.ref<T>` types to model bidirectional signal references
3. When `arcilator` runs the `LowerState` pass, it iterates over all module arguments
4. For each argument, it attempts to create a `RootInputOp` with `StateType::get(arg.getType())`
5. The `inout` port's `!llhd.ref<i1>` type is passed to `StateType::get()`
6. The verification fails because `llhd.ref` has no known bit width mapping

### Triggering Pattern in source.sv

```systemverilog
module MixedPorts(
  input  logic clk,
  input  logic a,
  output logic b,
  inout  logic c    // <-- This inout port triggers the bug
);
  // ...
  assign c = c_drive;   // Inout port being driven
  // ...
endmodule
```

The key pattern that triggers this bug:
1. **Presence of `inout` port**: The module contains an `inout logic c` port
2. **Bidirectional usage**: The `inout` port `c` is both assigned (`assign c = c_drive`) and potentially read
3. **Arcilator processing**: When arcilator attempts to lower this module, it fails to handle the `!llhd.ref` type

## Why This Is A Bug

The `LowerState` pass in arcilator doesn't properly handle LLHD reference types that arise from bidirectional ports. The pass assumes all module arguments can be wrapped in `StateType`, but this assumption breaks when LLHD dialect types are present.

This represents a **missing type handling case** in the Arc dialect's interaction with the LLHD dialect.

## Potential Fixes

### Option 1: Filter out unsupported port types
Before creating `RootInputOp`, check if the argument type is compatible with `StateType`:

```cpp
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto type = arg.getType();
  // Skip LLHD ref types which cannot be represented as state
  if (isa<llhd::RefType>(type)) {
    // Emit warning or error gracefully
    continue;
  }
  // ...existing code...
}
```

### Option 2: Extend computeLLVMBitWidth() to handle RefType
Add support for `llhd::RefType` by treating it as a pointer:

```cpp
if (auto refType = dyn_cast<llhd::RefType>(type)) {
  // Pointers are typically 64 bits on common architectures
  return 64;
}
```

### Option 3: Emit a proper diagnostic
Instead of failing an assertion, emit a proper error diagnostic:

```cpp
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto stateType = StateType::getChecked(
      arg.getLoc(), arg.getType());
  if (!stateType) {
    return emitError(arg.getLoc()) 
        << "inout ports are not supported by arcilator";
  }
  // ...
}
```

## Reproduction

```bash
circt-verilog --ir-hw source.sv | arcilator
```

Where `source.sv` contains a module with `inout` ports.

## Classification

| Category | Value |
|----------|-------|
| Bug Type | Missing type handling / Assertion failure |
| Severity | Medium (tool crash, not security issue) |
| Scope | Arcilator's LowerState pass |
| Root Cause | LLHD RefType not supported in StateType creation |
| Workaround | Avoid using `inout` ports when targeting arcilator |
