# [arcilator] Assertion failure when processing modules with inout ports

## Summary
Arcilator crashes with an assertion failure when processing SystemVerilog modules that contain `inout` (bidirectional) ports. The crash occurs in the `LowerState` pass when attempting to create a `StateType` for `!llhd.ref<i1>` type arguments.

## Environment
- **CIRCT Version**: 1.139.0 (commit 690366b6c)
- **LLVM Version**: 22.0.0git
- **Build Type**: Optimized build with assertions
- **Platform**: Linux x86_64

## Steps to Reproduce

### Minimal Test Case (bug.sv)
```systemverilog
// Minimal test case for arcilator crash with inout ports
// Bug: arcilator crashes on modules with inout (bidirectional) ports
// Root cause: LowerState pass cannot create StateType for llhd.ref type

module MinimalInout(inout logic c);
endmodule
```

### Commands
```bash
circt-verilog --ir-hw bug.sv | arcilator
```

### Generated IR
```mlir
module {
  hw.module @MinimalInout(in %c : !llhd.ref<i1>) {
    hw.output
  }
}
```

## Error Message
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: .../mlir/include/mlir/IR/StorageUniquerSupport.h:180: Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

## Stack Trace (Key Frames)
```
#12 circt::arc::StateType::get(mlir::Type) ArcTypes.cpp.inc:108
#13 (anonymous namespace)::ModuleLowering::run() LowerState.cpp:219
#14 (anonymous namespace)::LowerStatePass::runOnOperation() LowerState.cpp:1198
```

## Root Cause Analysis

The crash occurs in `lib/Dialect/Arc/Transforms/LowerState.cpp:219`:

```cpp
// Allocate storage for the inputs.
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(arg.getType()), name, storageArg);  // ← CRASH
  allocatedInputs.push_back(state);
}
```

The code iterates over **all** module arguments and attempts to create a `StateType` for each. When the argument type is `!llhd.ref<i1>` (generated from an `inout` port), `StateType::get()` fails because:

1. `StateType::verify()` calls `computeLLVMBitWidth(innerType)`
2. `computeLLVMBitWidth()` only handles: `seq::ClockType`, `IntegerType`, `hw::ArrayType`, `hw::StructType`
3. `llhd::RefType` is not handled, so `nullopt` is returned
4. Assertion fires in debug build

## Suggested Fix

### Option 1: Reject unsupported types with proper error (Recommended)
```cpp
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  if (isa<llhd::RefType>(arg.getType())) {
    return moduleOp.emitError()
        << "inout ports (llhd.ref type) are not supported by arcilator";
  }
  // ... existing code ...
}
```

### Option 2: Add llhd::RefType to computeLLVMBitWidth
```cpp
if (auto refType = dyn_cast<llhd::RefType>(type))
  return computeLLVMBitWidth(refType.getType());
```
Note: This may not be semantically correct since references aren't values.

## Cross-Tool Validation
The test case is valid SystemVerilog, accepted by:
- ✅ Verilator 5.022
- ✅ Icarus Verilog 548010e
- ✅ Slang 9.1.0

## Related
- Issue #8825 discusses `!llhd.ref<T>` type design (feature request, not crash)
