# Root Cause Analysis: Arc LowerState Crash on InOut Ports

## Summary

The arcilator crashes with an assertion failure when processing a SystemVerilog module containing `inout` ports. The `LowerState` pass in the Arc dialect fails to handle `llhd::RefType` (used for bidirectional ports), causing `StateType::get()` to fail its type validation.

## Crash Details

### Error Message
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
```

### Assertion Location
- **File**: `lib/Dialect/Arc/Transforms/LowerState.cpp:219`
- **Function**: `ModuleLowering::run()`
- **Assertion**: `StateType::verifyInvariants` via `StateType::get()`

### Stack Trace (Key Frames)
1. `StateType::get(mlir::Type)` - ArcTypes.cpp.inc:108
2. `ModuleLowering::run()` - LowerState.cpp:219
3. `LowerStatePass::runOnOperation()` - LowerState.cpp:1198

## Test Case Analysis

### Source Code (source.sv)
```systemverilog
module MixedPorts(input logic a, output logic b, inout wire c);
  logic [3:0] temp_reg;
  
  always_comb begin
    temp_reg = 4'b0;
    for(int i=0; i<4; i++) begin
      temp_reg[i] = a;
    end
    b = temp_reg[0];
  end
endmodule
```

### Triggering Construct
- **`inout wire c`**: Bidirectional port declaration
- The `inout` port gets lowered to `!llhd.ref<i1>` type in the IR

## Root Cause

### Technical Analysis

1. **Port Type Conversion**: When `circt-verilog --ir-hw` processes the SystemVerilog module, the `inout` port is converted to use `llhd::RefType` to represent the bidirectional signal reference.

2. **LowerState Pass Behavior** (LowerState.cpp:214-219):
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
   The pass iterates over **all** block arguments and unconditionally calls `StateType::get(arg.getType())`.

3. **StateType Validation** (ArcTypes.cpp:79-85):
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

4. **computeLLVMBitWidth Limitation** (ArcTypes.cpp:24-50):
   The function only handles:
   - `seq::ClockType`
   - `IntegerType`
   - `hw::ArrayType`
   - `hw::StructType`
   
   It does **not** handle `llhd::RefType`, so it returns `std::nullopt`, causing the assertion to fail.

### Why This Happens

The Arc dialect and arcilator are designed for **cycle-based simulation**, which doesn't naturally support bidirectional signals. The `llhd::RefType` represents a signal reference for LLHD's **discrete-event simulation** model, which is fundamentally different from Arc's simulation model.

The `LowerState` pass assumes all module arguments can be converted to `StateType`, but this assumption breaks when `inout` ports (represented as `llhd::RefType`) are present.

## Impact

- **Severity**: High - Causes compiler crash (assertion failure)
- **Scope**: Any SystemVerilog module with `inout` ports processed through arcilator
- **User Impact**: Cannot simulate designs with bidirectional ports using arcilator

## Potential Fixes

1. **Filter out unsupported types**: Skip `llhd::RefType` arguments in `ModuleLowering::run()` and emit a proper error diagnostic instead of crashing.

2. **Add llhd::RefType support**: Extend `computeLLVMBitWidth()` to handle `llhd::RefType` by extracting the nested type's bit width.

3. **Early validation**: Add a pass before `LowerState` that validates the module doesn't contain unsupported constructs and emits a clean error message.

## Related Issues

This appears to be related to the broader challenge of supporting LLHD constructs in the Arc dialect pipeline. Similar issues exist with other LLHD-specific operations that don't have Arc equivalents.

## Keywords

- arcilator
- LowerState
- StateType
- llhd.ref
- inout port
- assertion failure
- computeLLVMBitWidth
