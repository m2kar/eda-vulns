# Root Cause Analysis Report

## Executive Summary

The arcilator tool crashes when processing a SystemVerilog module containing an `inout` port. The `LowerState` pass attempts to create `arc::StateType` for the inout port's argument type (`!llhd.ref<i1>`), but `StateType` cannot handle reference types because they don't have a known bit width.

## Crash Context
- **Tool/Command**: `circt-verilog --ir-hw | arcilator`
- **Dialect**: SV → HW → Arc
- **Failing Pass**: `arc-lower-state` (LowerStatePass)
- **Crash Type**: Assertion failure in `StateType::get()`

## Error Analysis

### Assertion/Error Message
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: .../mlir/include/mlir/IR/StorageUniquerSupport.h:180: 
static ConcreteT mlir::detail::StorageUserBase<...>::get(...): 
Assertion `succeeded( ConcreteT::verifyInvariants(...))' failed.
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
module MixedPorts(
  input logic a,
  output logic b,
  inout logic c,      // <-- PROBLEM: inout port
  input logic clk
);
  logic c_drive;
  always_ff @(posedge clk) begin
    b <= a;
    c_drive <= a;
  end
  assign c = c_drive;
endmodule
```

The test case defines a module with **mixed port directions**: `input`, `output`, and critically an `inout` port.

### Key Constructs
- **`inout logic c`**: Bidirectional port - this is the trigger for the crash
- **`always_ff` block**: Sequential logic with clock
- **`assign c = c_drive`**: Drive assignment to inout port

### Potentially Problematic Patterns
The `inout` port is fundamentally incompatible with the Arc dialect's simulation model. When `circt-verilog --ir-hw` processes this, it converts the inout port to an `llhd.ref<i1>` type (LLHD's signal reference type) rather than a plain integer type.

## CIRCT Source Analysis

### Crash Location
**File**: `lib/Dialect/Arc/Transforms/LowerState.cpp`
**Function**: `ModuleLowering::run()`
**Line**: ~219

### Code Context
```cpp
// LowerState.cpp:214-220
// Allocate storage for the inputs.
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(arg.getType()), name, storageArg);
  //                      ^^^^^^^^^^^^^^^^^^^^^^^^^^
  //                      CRASH: arg.getType() is !llhd.ref<i1>
  allocatedInputs.push_back(state);
}
```

### StateType Verification
```cpp
// ArcTypes.cpp:78-84
LogicalResult
StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  Type innerType) {
  if (!computeLLVMBitWidth(innerType))  // <-- llhd.ref fails this check
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

The `computeLLVMBitWidth()` function only handles:
- `seq::ClockType` → 1 bit
- `IntegerType` → width
- `hw::ArrayType` → computed
- `hw::StructType` → computed
- **Everything else returns `{}`** (no valid width)

Since `llhd::RefType` is not handled, `computeLLVMBitWidth()` returns `{}`, causing verification to fail.

### Processing Path
1. **circt-verilog**: Parses SystemVerilog, sees `inout logic c`
2. **MooreToCore**: Converts Moore dialect to HW/LLHD dialect
   - Inout ports become block arguments with `llhd.ref<T>` type
3. **arcilator (LowerStatePass)**: Iterates over module block arguments
4. **StateType::get()**: Called with `llhd.ref<i1>` type
5. **CRASH**: Verification fails because llhd.ref has no known bit width

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: Missing support for `inout` ports in the Arc dialect's LowerState pass

**Evidence**:
- `ModelOp::verify()` explicitly rejects inout ports: `"inout ports are not supported"` (ArcOps.cpp:336)
- LowerState.cpp blindly iterates all block arguments without checking port direction
- `StateType` cannot wrap `llhd.ref` types

**Mechanism**:
1. `inout` ports are represented as `llhd.ref<T>` in HW IR
2. LowerState pass doesn't filter or reject inout ports early
3. Attempt to create `StateType::get(llhd.ref<i1>)` triggers verification failure

### Hypothesis 2 (Medium Confidence)
**Cause**: Missing conversion/elimination pass for inout ports before arcilator

**Evidence**:
- There exists `HWEliminateInOutPorts` pass in SV dialect
- arcilator pipeline may not include this pass
- The error occurs late in lowering, suggesting early validation is missing

**Mechanism**:
The arcilator pipeline should either:
1. Run `hw-eliminate-inout-ports` before lowering, OR
2. Emit a clear error early when inout ports are detected

### Hypothesis 3 (Lower Confidence)
**Cause**: Type conversion issue in MooreToCore for inout signals

**Evidence**:
- MooreToCore.cpp uses `llhd::RefType` for variable/signal types
- Inout ports may be incorrectly preserved as llhd.ref instead of converted

**Mechanism**:
The frontend may need to transform inout ports differently for arcilator compatibility.

## Suggested Fix Directions

1. **Add early validation in LowerStatePass** (Recommended):
   ```cpp
   // In ModuleLowering::run(), before processing arguments:
   for (auto port : moduleOp.getIo().getPorts()) {
     if (port.dir == hw::ModulePort::Direction::InOut) {
       moduleOp.emitOpError("inout ports are not supported by arcilator");
       return failure();
     }
   }
   ```

2. **Extend StateType to handle or reject RefType gracefully**:
   ```cpp
   static std::optional<uint64_t> computeLLVMBitWidth(Type type) {
     // Add explicit handling for RefType
     if (isa<llhd::RefType>(type))
       return {};  // Already returns {}, but could add diagnostic
     // ...
   }
   ```

3. **Document unsupported features in arcilator**:
   - Add clear documentation that arcilator doesn't support inout/bidirectional ports
   - Suggest using `hw-eliminate-inout-ports` before arcilator if applicable

## Keywords for Issue Search
`arcilator` `inout` `StateType` `llhd.ref` `LowerState` `bit width` `assertion` `bidirectional` `port`

## Related Files to Investigate
- `lib/Dialect/Arc/Transforms/LowerState.cpp` - Crash site, needs early validation
- `lib/Dialect/Arc/ArcTypes.cpp` - StateType verification logic
- `lib/Dialect/Arc/ArcOps.cpp` - ModelOp already has inout rejection (line 336)
- `lib/Dialect/SV/Transforms/HWEliminateInOutPorts.cpp` - Potential preprocessing pass
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Inout port conversion
- `lib/Tools/arcilator/pipelines.cpp` - arcilator pass pipeline configuration
