# Root Cause Analysis Report

## Summary

**Crash Type:** Assertion Failure  
**Dialect:** Arc/LLHD  
**Component:** arcilator (LowerState pass)  
**Severity:** High (compiler crash)

## Error Details

### Error Message
```
state type must have a known bit width; got '!llhd.ref<i1>'
```

### Crash Location
- **File:** `lib/Dialect/Arc/Transforms/LowerState.cpp`
- **Line:** 219 (in `ModuleLowering::run()`)
- **Function:** `StateType::get()` called during state allocation

### Stack Trace Summary
```
#12 circt::arc::StateType::get(mlir::Type) - ArcTypes.cpp.inc:108
#13 ModuleLowering::run() - LowerState.cpp:219
#14 LowerStatePass::runOnOperation() - LowerState.cpp:1198
```

## Root Cause Analysis

### The Problem

The crash occurs when the `arcilator` tool attempts to lower a hardware module containing an **inout port** (bidirectional signal) to the Arc dialect's simulation model. The issue is in the `StateType::get()` function, which requires the inner type to have a **known bit width** that can be computed by `computeLLVMBitWidth()`.

### Technical Details

1. **Input SystemVerilog Code:**
   ```systemverilog
   module test_module(inout wire c, input logic a);
     logic [3:0] temp_reg;
     initial begin
       temp_reg = 4'b1010;
     end
     assign c = (a) ? temp_reg[0] : 1'bz;
   endmodule
   ```

2. **The Problematic Construct:**
   - The `inout wire c` port creates an LLHD `ref` type (`!llhd.ref<i1>`) in the IR
   - When the `LowerState` pass tries to allocate storage for module inputs, it calls `StateType::get(arg.getType())` for each argument
   - The `StateType::verify()` function calls `computeLLVMBitWidth()` to ensure the type has a known bit width

3. **Why `!llhd.ref<i1>` Fails:**
   The `computeLLVMBitWidth()` function in `lib/Dialect/Arc/ArcTypes.cpp` only handles these types:
   - `seq::ClockType` → returns 1
   - `IntegerType` → returns the integer width
   - `hw::ArrayType` → computed recursively
   - `hw::StructType` → computed recursively
   
   **It does NOT handle `llhd::RefType`**, so it returns `std::nullopt`, causing the assertion to fail.

4. **Crash Trigger Point:**
   In `LowerState.cpp` around line 219 (in `ModuleLowering::run()`):
   ```cpp
   // Allocate storage for the inputs.
   for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
     auto name = moduleOp.getArgName(arg.getArgNumber());
     auto state =
         RootInputOp::create(allocBuilder, arg.getLoc(),
                             StateType::get(arg.getType()), name, storageArg);
     //                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     //                      This call fails for !llhd.ref<i1>
     allocatedInputs.push_back(state);
   }
   ```

### Why This Is a Bug

1. **Inconsistent Type Handling:** The `circt-verilog --ir-hw` frontend successfully parses the inout port and represents it as `!llhd.ref<i1>`, but the downstream `arcilator` pass doesn't support this type.

2. **Missing Validation:** There's no early check to reject unsupported types before the assertion failure occurs. The tool should emit a proper error diagnostic instead of crashing.

3. **Incomplete LLHD Type Support:** The Arc dialect's `computeLLVMBitWidth()` function doesn't handle LLHD reference types, which are valid in the IR produced by the frontend.

## Potential Fixes

### Option 1: Add LLHD RefType Support
Extend `computeLLVMBitWidth()` in `ArcTypes.cpp` to handle `llhd::RefType`:
```cpp
if (auto refType = dyn_cast<llhd::RefType>(type))
  return computeLLVMBitWidth(refType.getNestedType());
```

### Option 2: Early Rejection with Diagnostic
Add validation in `LowerState.cpp` to check for unsupported port types and emit a proper error:
```cpp
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  if (isa<llhd::RefType>(arg.getType())) {
    return moduleOp.emitError() 
        << "inout ports are not supported by arcilator";
  }
  // ... rest of allocation code
}
```

### Option 3: Frontend Restriction
Prevent `circt-verilog` from generating `!llhd.ref` types when targeting arcilator output, or transform them appropriately before the LowerState pass.

## Related Components

- **circt-verilog:** Frontend that generates the IR with `!llhd.ref` types
- **Arc Dialect:** Target dialect for simulation
- **LLHD Dialect:** Source dialect for signal references
- **LowerState Pass:** Transformation pass that triggers the crash

## Keywords for Duplicate Detection

- StateType
- bit width
- llhd.ref
- arcilator
- LowerState
- inout
- computeLLVMBitWidth
- assertion failure
- state type must have a known bit width

## Reproduction Command

```bash
circt-verilog --ir-hw source.sv | arcilator
```

## Environment

- **Tool Version:** CIRCT 1.139.0
- **Crash Tool:** arcilator
- **Input:** SystemVerilog with inout port
