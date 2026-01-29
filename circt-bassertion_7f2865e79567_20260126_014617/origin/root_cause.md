# Root Cause Analysis Report

## Executive Summary
The crash occurs because arcilator's LowerState pass attempts to create an Arc StateType from an LLHD reference type (!llhd.ref<i1>), which is invalid. The test case's inout wire port triggers this by creating LLHD references that aren't properly handled during state lowering.

## Crash Context
- Tool: arcilator
- Dialect: Arc (from HW/LLHD)
- Failing Pass: LowerState
- Crash Type: Assertion failure

## Error Analysis

### Assertion Message
```
state type must have a known bit width; got '!llhd.ref<i1>'
```

### Key Stack Frames
```
#12 circt::arc::StateType::get(mlir::Type)
#13 ModuleLowering::run() LowerState.cpp:219
#15 LowerStatePass::runOnOperation()
```

### Crash Location
- **File**: `/home/zhiqing/edazz/circt/lib/Dialect/Arc/Transforms/LowerState.cpp`
- **Function**: `ModuleLowering::run()`
- **Line**: 219

```cpp
// Code context at crash site:
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(arg.getType()), name, storageArg);  // <-- CRASH HERE
  allocatedInputs.push_back(state);
}
```

### Type Verification
The error originates from `ArcTypes.cpp:83-85`:

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
- ClockType
- IntegerType
- hw::ArrayType
- hw::StructType

LLHD ref types (e.g., `!llhd.ref<i1>`) are not in this list, causing the verification failure.

## Test Case Analysis

### Code Summary
Module with mixed port types including an inout wire port that is read in an always_ff block.

### SystemVerilog Code
```systemverilog
module MixedPorts(
  input logic a,
  output logic b,
  inout wire c,        // <-- PROBLEMATIC PORT
  input logic clk
);

  logic [3:0] temp_reg;

  assign b = 4'd2;

  // Always block using inout port and clock
  always_ff @(posedge clk) begin
    temp_reg <= c;     // Reading from inout creates LLHD ref type
  end

  // Bidirectional assignment for inout port
  assign c = a ? temp_reg : 4'bz;

endmodule
```

### Key Constructs
- inout wire `c` (bidirectional port)
- always_ff block reading from inout port
- Tri-state assignment with `4'bz`
- Sized decimal constant assignment `4'd2`

### Problematic Pattern
Reading from an **inout port in sequential logic** (`temp_reg <= c`) causes circt-verilog to generate LLHD reference types for module arguments. These refs are then passed to the Arc lowering, which expects types with known bit widths.

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence) - Primary Cause
**Cause**: LowerState pass doesn't handle LLHD reference types when creating Arc states from module arguments

**Evidence**:
1. **Explicit Error**: Error message clearly states `"state type must have a known bit width; got '!llhd.ref<i1>'"`
2. **Stack Trace**: Confirms crash at `StateType::get(arg.getType())` in line 219
3. **Test Pattern**: inout port `c` read in always_ff creates module arguments of type `!llhd.ref<i1>`
4. **Missing Type Support**: `computeLLVMBitWidth()` doesn't handle LLHD ref types

**Mechanism**:
```
1. circt-verilog converts inout wire c to LLHD signal reference (!llhd.ref<i1>)
2. The module argument c has type !llhd.ref<i1>
3. LowerState pass iterates over all module arguments
4. For each argument, it tries: StateType::get(arg.getType())
5. This calls StateType::verify(innerType) where innerType = !llhd.ref<i1>
6. computeLLVMBitWidth() returns {} for ref types
7. Verification fails with assertion
```

### Hypothesis 2 (Medium Confidence) - Design Gap
**Cause**: No mechanism to convert LLHD ref types to their underlying element types before state creation

**Evidence**:
- LLHD ref types are valid in the IR but incompatible with Arc's state model
- Other ports (input logic, output logic) work fine because they have primitive types
- The ref type is an abstraction that should be dereferenced during lowering

**Mechanism**:
- Arc dialect expects primitive types (i1, i32, arrays, structs)
- LLHD adds a layer of indirection with ref types
- The lowering should extract the element type before creating states

### Hypothesis 3 (Low Confidence) - Toolchain Issue
**Cause**: circt-verilog shouldn't produce LLHD ref types for inout ports in this context

**Evidence**:
- inout ports are common in Verilog/SystemVerilog
- arcilator is a simulator, so it should handle them
- The error suggests a fundamental incompatibility in the design

**Mechanism**:
- circt-verilog might incorrectly infer ref types for inout ports
- It should probably use the underlying element type instead
- This would require changing the HW->LLHD lowering logic

## Comparison with Similar Issues

### Known LLHD Behavior
LLHD uses reference types to model connections:
- `!llhd.ref<i1>` - reference to an i1 value
- `!llhd.ref<i32>` - reference to an i32 value

These refs are used internally for signal connections but shouldn't appear in the final state representation.

### Previous Fix Examples
Similar issues were fixed in other dialect conversions:
1. **Moore to LLHD**: Array indexing issues fixed by proper type extraction
2. **HW to LLHD**: Various type mapping improvements

The fix pattern typically involves:
1. Detecting unsupported types before state creation
2. Converting unsupported types to supported ones (dereferencing, unwrapping)
3. Providing clear error messages for unsupported constructs

## Suggested Fix Directions

### Option 1: Type Conversion in LowerState (Recommended)
**Implementation**: Modify `ModuleLowering::run()` to handle ref types

```cpp
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto argType = arg.getType();

  // Handle LLHD ref types - extract element type
  if (auto refType = dyn_cast<LLHDRefType>(argType)) {
    argType = refType.getElementType();
  }

  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(argType), name, storageArg);
  allocatedInputs.push_back(state);
}
```

**Pros**:
- Minimal code change
- Preserves existing semantics
- Handles all ref type variants

**Cons**:
- Requires adding LLHD dialect header
- Might hide deeper type system issues

### Option 2: Early Type Validation
**Implementation**: Check argument types before creating states

```cpp
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto argType = arg.getType();

  // Validate type is supported for state lowering
  if (!StateType::verify(
        [&](InFlightDiagnostic &&diag) { return diag; }, argType).succeeded()) {
    return arg.emitError() << "Module argument of type "
                           << argType
                           << " cannot be used as Arc state input. "
                           << "LLHD ref types are not supported.";
  }

  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(argType), name, storageArg);
  allocatedInputs.push_back(state);
}
```

**Pros**:
- Better error message for users
- Prevents obscure assertion failures
- Makes the limitation explicit

**Cons**:
- Doesn't actually fix the problem
- Just provides a friendlier error

### Option 3: Fix in circt-verilog (Deep Fix)
**Implementation**: Prevent LLHD ref types from being created for inout ports

This would require modifying the HW-to-LLHD lowering to not generate refs for certain port types.

**Pros**:
- Addresses the root cause
- Prevents similar issues in other contexts

**Cons**:
- Large change scope
- Affects many lowering passes
- Higher risk of introducing new bugs

## Keywords for Issue Search
`arcilator` `LowerState` `inout` `llhd.ref` `StateType` `computeLLVMBitWidth` `type verification`

## Related Files
- `/home/zhiqing/edazz/circt/lib/Dialect/Arc/Transforms/LowerState.cpp` - Crash location
- `/home/zhiqing/edazz/circt/lib/Dialect/Arc/ArcTypes.cpp` - StateType::verify() and computeLLVMBitWidth()
- `/home/zhiqing/edazz/circt/lib/Conversion/HWToLLHD/` - HW to LLHD lowering (if exists)
- `/home/zhiqing/edazz/circt/lib/Dialect/LLHD/IR/` - LLHD type definitions

## Additional Observations

### Why Only This Test Case?
The test case specifically uses:
1. An inout port (less common than input/output)
2. Reads the inout in sequential logic (always_ff)
3. No other ports have this issue

Simple tests without inout ports work fine:
```systemverilog
module SimpleTest(input logic a, input logic clk);
  always_ff @(posedge clk) begin
    // No crash because a is already an i1
  end
endmodule
```

### Compiler Version
Built from source (circt-src directory present)
CIRCT version: 1.139.0 (from error log)

### Impact
- **Severity**: Medium - crashes the simulator on valid SystemVerilog
- **Frequency**: Low - only affects code with inout ports in sequential logic
- **Scope**: All arcilator usage with inout ports

## Conclusion
The crash is caused by an incompatibility between LLHD's reference type system and Arc's state type requirements. The LowerState pass attempts to create a StateType from an LLHD ref type without first extracting the underlying element type. The recommended fix is to handle ref types in the LowerState pass by dereferencing them before state creation.
