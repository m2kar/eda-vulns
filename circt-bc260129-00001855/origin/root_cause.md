# CIRCT Crash Root Cause Analysis Report

## 1. Crash Summary

| Item | Value |
|------|-------|
| **Testcase ID** | 260129-00001855 |
| **Crash Type** | Assertion Failure |
| **Tool** | arcilator |
| **Dialect** | Arc (Arc dialect's LowerState pass) |
| **Severity** | High |

## 2. Error Message

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: .../mlir/include/mlir/IR/StorageUniquerSupport.h:180: 
Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

## 3. Crash Location

### Stack Trace Analysis

The crash occurs in the following call chain:

1. **`main`** → `executeArcilator` (arcilator.cpp:697)
2. **`PassManager::run`** → executes the Arc pipeline
3. **`LowerStatePass::runOnOperation`** (LowerState.cpp:1198)
4. **`ModuleLowering::run`** (LowerState.cpp:219)
5. **`StateType::get(mlir::Type)`** (ArcTypes.cpp.inc:108)
6. **`StorageUserBase::get`** → Assertion failure in verifyInvariants

### Key Source Location

**File:** `lib/Dialect/Arc/ArcTypes.cpp`

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

**File:** `lib/Dialect/Arc/Transforms/LowerState.cpp` (Line ~219)

```cpp
// In ModuleLowering::run():
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(arg.getType()), name, storageArg);
  allocatedInputs.push_back(state);
}
```

## 4. Root Cause Analysis

### 4.1 Problem Description

The crash occurs when the `arcilator` tool attempts to lower a hardware module that contains **inout (bidirectional) ports**. The Arc dialect's `LowerState` pass tries to create `StateType` for all module arguments, but fails when encountering an `!llhd.ref<i1>` type.

### 4.2 Technical Details

1. **Input Processing:** When `circt-verilog` parses the SystemVerilog source with inout ports (`inout logic port_a, port_b`), it generates an IR representation where bidirectional ports are represented using `!llhd.ref<T>` types.

2. **Type Incompatibility:** The `StateType::verify()` function calls `computeLLVMBitWidth()` to determine the bit width of the inner type. This function only handles:
   - `seq::ClockType` → returns 1
   - `IntegerType` → returns width
   - `hw::ArrayType` → computes element width × count
   - `hw::StructType` → computes struct layout

3. **Missing Handler:** The `computeLLVMBitWidth()` function does **not** handle `llhd::RefType`, causing it to return `std::nullopt`, which triggers the verification failure.

4. **Assertion Trigger:** When `StateType::get()` is called with an `!llhd.ref<i1>` type, the verification fails, and the MLIR infrastructure triggers an assertion failure because `verifyInvariants` did not succeed.

### 4.3 Triggering Conditions

The bug is triggered by:
1. A SystemVerilog module with **inout (bidirectional) ports**
2. Processing through the `circt-verilog --ir-hw` pipeline
3. Passing the output to `arcilator` for simulation compilation

### 4.4 Code Flow

```
SystemVerilog source (with inout ports)
    ↓
circt-verilog --ir-hw
    ↓
HW IR with llhd.ref types for inout ports
    ↓
arcilator (LowerState pass)
    ↓
ModuleLowering::run()
    ↓
StateType::get(llhd.ref<i1>)  ← CRASH HERE
    ↓
StateType::verify() fails
    ↓
Assertion failure
```

## 5. Test Case Analysis

### 5.1 Source Code (`source.sv`)

```systemverilog
module test_module (
  input logic clk,
  output logic out,
  inout logic port_a,    // ← Problematic: bidirectional port
  inout logic port_b     // ← Problematic: bidirectional port
);
  logic sig;
  
  assign port_a = sig;
  assign port_b = sig;
  
  always @(posedge clk) begin
    sig = out;
  end
  
  always_comb begin
    out = ~sig;
  end
endmodule
```

### 5.2 Key Problematic Construct

The **`inout logic`** port declarations are the root cause. When lowered to HW IR:
- `input logic clk` → `i1` type (or `seq::ClockType`)
- `output logic out` → `i1` type
- `inout logic port_a` → `!llhd.ref<i1>` type (reference type for bidirectionality)

The `llhd.ref<T>` type represents a reference to a signal that can be both read and written, which is semantically correct for bidirectional ports but is not supported by the Arc dialect's state allocation mechanism.

## 6. Classification

| Aspect | Classification |
|--------|----------------|
| **Bug Type** | Missing Type Handler / Incomplete Lowering Support |
| **Component** | Arc Dialect - LowerState Pass |
| **Impact** | Prevents simulation of designs with bidirectional ports |
| **Reproducibility** | 100% reproducible |

## 7. Suggested Fix

### Option 1: Add llhd.ref support to computeLLVMBitWidth

```cpp
static std::optional<uint64_t> computeLLVMBitWidth(Type type) {
  // ... existing handlers ...
  
  // Add handler for llhd::RefType
  if (auto refType = dyn_cast<llhd::RefType>(type))
    return computeLLVMBitWidth(refType.getNestedType());
  
  // We don't know anything about any other types.
  return {};
}
```

### Option 2: Handle inout ports specially in LowerState

Bidirectional ports may require special handling in the `LowerState` pass rather than being treated as simple state storage.

### Option 3: Early validation and graceful error

Add a pre-pass check to detect unsupported constructs and emit a clear diagnostic instead of an assertion failure.

## 8. Reproduction Command

```bash
circt-verilog --ir-hw source.sv | arcilator
```

## 9. Related Information

- **CIRCT Version:** 1.139.0
- **Affected Pass:** `arc-lower-state`
- **Key Files:**
  - `lib/Dialect/Arc/Transforms/LowerState.cpp`
  - `lib/Dialect/Arc/ArcTypes.cpp`
  - `include/circt/Dialect/Arc/ArcTypes.td`
