# Root Cause Analysis Report

## Error Summary
- **Crash type**: Assertion failure
- **Error message**:
  - `state type must have a known bit width; got '!llhd.ref<i1>'`
  - Assertion from `mlir::detail::StorageUserBase<circt::arc::StateType,...>::get` in `StorageUniquerSupport.h:180`
- **Tool**: `arcilator`
- **Phase/Component**: Arc dialect `LowerState` pass (`LowerStatePass::runOnOperation`)

Evidence:
- error.txt lines 7–9 show the error message and assertion.
- Stack trace shows `circt::arc::StateType::get` and `ModuleLowering::run` in `lib/Dialect/Arc/Transforms/LowerState.cpp:219` (error.txt lines 24–27).

## Stack Trace Analysis
Key frames:
1. `circt::arc::StateType::get(mlir::Type)`
   - Source: `circt/Dialect/Arc/ArcTypes.cpp.inc:108` (via stack)
2. `(anonymous namespace)::ModuleLowering::run()`
   - Source: `lib/Dialect/Arc/Transforms/LowerState.cpp:219`
3. `LowerStatePass::runOnOperation()`
   - Source: `lib/Dialect/Arc/Transforms/LowerState.cpp:1198`

The assertion is triggered during construction of an `arc::StateType` from a type that does not have a known LLVM storage bit width.

## Test Case Analysis (source.sv)
SystemVerilog constructs used:
- `inout wire c` port (tri-state capable port)
- Continuous assignments:
  - `assign b = a;`
  - `assign c = a ? 1'bz : 1'b0;` (explicit high-impedance `'z'`)
- `always_comb` block
- `always_ff @(posedge clk)` with nonblocking assignment

Notable/edge-case patterns:
- **Tri-state on inout**: `c` is driven by a conditional that outputs `'z'` and `'0'`.
- The presence of an `inout` port often lowers to a **reference-like** type in intermediate representations.

Evidence:
- `source.sv` lines 1–20 show `inout wire c` and `'z'` assignment.
- Error message explicitly mentions `!llhd.ref<i1>` as the type that lacks a known bit width.

## Source Code Analysis (CIRCT)
Relevant code:

### 1) ArcTypes.cpp
`computeLLVMBitWidth` only supports:
- `seq::ClockType`
- `IntegerType`
- `hw::ArrayType`
- `hw::StructType`

```cpp
// circt-src/lib/Dialect/Arc/ArcTypes.cpp
static std::optional<uint64_t> computeLLVMBitWidth(Type type) {
  if (isa<seq::ClockType>(type)) return 1;
  if (auto intType = dyn_cast<IntegerType>(type)) return intType.getWidth();
  if (auto arrayType = dyn_cast<hw::ArrayType>(type)) { ... }
  if (auto structType = dyn_cast<hw::StructType>(type)) { ... }
  return {};
}

LogicalResult StateType::verify(..., Type innerType) {
  if (!computeLLVMBitWidth(innerType))
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

This rejects `llhd.ref` types (no handling). (Evidence: ArcTypes.cpp lines 29–85.)

### 2) LowerState.cpp
In `ModuleLowering::run`, the lowering allocates storage for module arguments:

```cpp
// circt-src/lib/Dialect/Arc/Transforms/LowerState.cpp
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state = RootInputOp::create(allocBuilder, arg.getLoc(),
                                   StateType::get(arg.getType()),
                                   name, storageArg);
  allocatedInputs.push_back(state);
}
```

This unconditionally creates `StateType::get(arg.getType())` for **all** module arguments (line ~219).

If any argument has type `!llhd.ref<i1>`, `StateType::verify` fails because `llhd.ref` does not have a known bit width. (Evidence: error message and stack trace.)

## Root Cause Hypothesis
**What exactly causes the crash**:
- `LowerState` attempts to allocate state for **every module argument** by wrapping its type in `arc::StateType`.
- When the argument type is `!llhd.ref<i1>`, `arc::StateType::verify` rejects it because `computeLLVMBitWidth` does not support `llhd.ref`.
- This triggers an assertion in `StorageUniquerSupport.h` during `StateType::get`.

**Why this input triggers it**:
- The test case uses an **`inout wire`** (`c`) with a tri-state assignment (`'z'`), which in lowerings frequently maps to a **reference type** in LLHD (`llhd.ref<i1>`).
- The error message explicitly shows that a `llhd.ref<i1>` reached `arc::StateType::verify`.
- Therefore, the module argument corresponding to `inout` is likely lowered to `llhd.ref<i1>` and hits the unsupported type path in `LowerState`.

**Underlying bug in CIRCT**:
- `LowerState` does not guard against or transform reference-like types before wrapping them in `arc::StateType`.
- `arc::StateType` verification lacks support for `llhd.ref` types and asserts instead of handling/diagnosing gracefully.

## Affected Components/Passes
- **Arc dialect**: `StateType` verification (`ArcTypes.cpp`)
- **Arc lowering pass**: `LowerStatePass` (`lib/Dialect/Arc/Transforms/LowerState.cpp`)
- **Tool**: `arcilator` pipeline (invokes the pass)
