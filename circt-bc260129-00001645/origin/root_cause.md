# CIRCT Crash Root Cause Analysis

## 1. Crash Summary

| Field | Value |
|-------|-------|
| **Testcase ID** | 260129-00001645 |
| **Crash Type** | Assertion Failure |
| **Tool** | arcilator |
| **Dialect** | Arc (with LLHD type interaction) |
| **Pass** | LowerState |
| **Severity** | High |

## 2. Error Context

### 2.1 Error Message
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
```

### 2.2 Assertion Failure
```cpp
circt/arc/StateType::get(mlir::Type):
  Assertion `succeeded(ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### 2.3 Stack Trace Key Frames
1. `StateType::get(mlir::Type)` - ArcTypes.cpp.inc:108
2. `ModuleLowering::run()` - LowerState.cpp:219
3. `LowerStatePass::runOnOperation()` - LowerState.cpp:1198

## 3. Root Cause Analysis

### 3.1 Problem Description

The crash occurs in the `arcilator` tool during the **LowerState** pass when attempting to create a `StateType` with an unsupported inner type `!llhd.ref<i1>`.

### 3.2 Technical Details

#### StateType Verification Constraint
The `StateType` in the Arc dialect has a verification constraint that requires the inner type to have a **known bit width**. This is defined in `lib/Dialect/Arc/ArcTypes.cpp`:

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

#### Unsupported Type: `!llhd.ref<i1>`
The `computeLLVMBitWidth()` function only handles the following types:
- `seq::ClockType`
- `IntegerType`
- `hw::ArrayType`
- `hw::StructType`

The `!llhd.ref<i1>` type (LLHD reference type) is **not** in this list, causing `computeLLVMBitWidth()` to return `std::nullopt`, which triggers the verification failure.

#### Crash Location
The crash occurs in `LowerState.cpp` around line 219, in the `ModuleLowering::run()` function. This is where module inputs are being allocated:

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

When `arg.getType()` is `!llhd.ref<i1>` (from the inout port `c`), calling `StateType::get(arg.getType())` triggers the assertion failure.

### 3.3 Source Code Trigger

The test case `source.sv` contains an **inout port**:
```systemverilog
module MixedPorts(
  input  logic        clk,
  input  logic signed [15:0] a,
  input  logic        [15:0] b,
  output logic        out_b,
  inout  logic        c       // <-- This is the problematic port
);
```

The `inout` port gets lowered to `!llhd.ref<i1>` type in the IR, which is a reference type used by LLHD dialect for bidirectional signals. However, the Arc dialect's `StateType` cannot handle this type.

### 3.4 Root Cause Summary

**The LowerState pass in arcilator does not properly handle `inout` (bidirectional) ports.** When a module contains an `inout` port, it gets represented as an `!llhd.ref<T>` type. The pass attempts to allocate state storage for all module arguments without checking if the type is supported by `StateType`. Since `!llhd.ref<T>` is not a type with a known bit width (as recognized by the Arc dialect), the assertion fails.

## 4. Bug Classification

| Aspect | Classification |
|--------|----------------|
| **Bug Type** | Missing Type Handling / Unsupported Feature |
| **Component** | Arc Dialect - LowerState Pass |
| **Impact** | Compilation failure for designs with inout ports |
| **Workaround** | Avoid using `inout` ports when targeting arcilator |

## 5. Suggested Fix

The fix could be one of the following approaches:

### Option A: Reject unsupported types gracefully
Add explicit type checking before calling `StateType::get()`:
```cpp
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto type = arg.getType();
  if (isa<llhd::RefType>(type)) {
    return moduleOp.emitOpError()
        << "inout ports are not supported by arcilator";
  }
  // ... existing code
}
```

### Option B: Support LLHD reference types
Extend `computeLLVMBitWidth()` to handle `llhd::RefType`:
```cpp
if (auto refType = dyn_cast<llhd::RefType>(type)) {
  return computeLLVMBitWidth(refType.getNestedType());
}
```

### Option C: Convert inout to input/output pair
Add a preprocessing pass that converts inout ports to separate input and output ports before lowering.

## 6. Reproduction

### 6.1 Command
```bash
circt-verilog --ir-hw source.sv | arcilator
```

### 6.2 Minimal Test Case
```systemverilog
module Test(inout logic c);
endmodule
```

## 7. References

- Source file: `lib/Dialect/Arc/Transforms/LowerState.cpp`
- Type verification: `lib/Dialect/Arc/ArcTypes.cpp`
- Type definition: `include/circt/Dialect/Arc/ArcTypes.td`
