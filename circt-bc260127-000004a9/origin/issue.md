# [circt-verilog][arcilator] Assertion failure when processing inout ports - StateType verification fails

## Summary

When using `arcilator` to process SystemVerilog code containing `inout` (bidirectional) ports, the tool crashes with an assertion failure in the `LowerState` pass. The crash occurs because `StateType::verify()` fails when trying to validate `llhd::RefType` (which represents inout ports), as `computeLLVMBitWidth()` cannot compute a bit width for reference types.

## Reproduction

### Minimal Test Case

```systemverilog
module M(inout x);
endmodule
```

### Command

```bash
circt-verilog --ir-hw bug.sv | arcilator
```

### Expected Behavior

The tool should either:
1. Successfully process the inout port, or
2. Emit a meaningful diagnostic error message

### Actual Behavior

The tool crashes with an assertion failure:

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/IR/StorageUniquerSupport.h:180: static ConcreteT mlir::detail::StorageUserBase<circt::arc::StateType, mlir::Type, circt::arc::detail::StateTypeStorage, mlir::detail::TypeUniquer>::get(MLIRContext *, Args &&...) [ConcreteT = circt::arc::StateType, BaseT = mlir::Type, StorageT = circt::arc::detail::StateTypeStorage, UniquerT = mlir::detail::TypeUniquer, Traits = <>, Args = <mlir::Type &>]: Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

## Environment

- **CIRCT Version**: 1.139.0
- **LLVM Version**: 22.0.0git
- **Tools**: circt-verilog, arcilator
- **Dialects**: Arc, LLHD, HW

## Stack Trace (Key Frames)

```
#12 circt::arc::StateType::get(mlir::Type)
    ArcTypes.cpp.inc:108:3

#13 (anonymous namespace)::ModuleLowering::run()
    LowerState.cpp:219:66

#14 (anonymous namespace)::LowerStatePass::runOnOperation()
    LowerState.cpp:1198:41

#15 mlir::detail::OpToOpPassAdaptor::run(...)
    Pass.cpp:619:17
```

## Root Cause Analysis

### Problem Chain

1. **Verilog to LLHD Lowering**
   - When `circt-verilog --ir-hw` processes the `inout x` port, it generates an `llhd::RefType` (`!llhd.ref<i1>`) to represent the reference to the inout signal.

2. **arcilator LowerState Pass** (file: `lib/Dialect/Arc/Transforms/LowerState.cpp`, line 219)
   - The `ModuleLowering::run()` function attempts to create an `arc::StateType` for module inputs/outputs:
   ```cpp
   auto state = RootInputOp::create(allocBuilder, arg.getLoc(),
                   StateType::get(arg.getType()), name, storageArg);
   ```

3. **StateType Verification Failure** (file: `lib/Dialect/Arc/ArcTypes.cpp`)
   - `arc::StateType` requires an inner type with a known bit width.
   - The verification function calls `computeLLVMBitWidth(innerType)`:
   ```cpp
   LogicalResult StateType::verify(..., Type innerType) {
     if (!computeLLVMBitWidth(innerType))
       return emitError() << "state type must have a known bit width; got "
                          << innerType;
     return success();
   }
   ```

4. **Bit Width Computation Fails**
   - `computeLLVMBitWidth()` only supports:
     - `seq::ClockType`
     - `IntegerType`
     - `hw::ArrayType`
     - `hw::StructType`
   - It **cannot handle `llhd::RefType`**, returning `std::nullopt`.

5. **Assertion Triggered**
   - Since `verifyInvariants()` fails, MLIR's `StorageUniquerSupport.h:180` assertion triggers, causing the crash.

### Core Issue

The `LowerState` pass lacks proper type validation before attempting to create `StateType` objects. It assumes all types passed to `StateType::get()` are supported, but `llhd::RefType` (representing inout ports) is not.

## Affected Code Paths

- **File**: `lib/Dialect/Arc/Transforms/LowerState.cpp`
  - Line 219 in `ModuleLowering::run()`
  - Attempted creation of `StateType` from unsupported type

- **File**: `lib/Dialect/Arc/ArcTypes.cpp`
  - `StateType::verify()` function
  - Lacks support for `llhd::RefType` in `computeLLVMBitWidth()`

- **File**: `lib/Dialect/LLHD/IR/LLHDTypes.td`
  - `RefType` definition for inout references

## Related Issues

- **Most Similar**: [#9467](https://github.com/llvm/circt/issues/9467) - arcilator fails to lower `llhd.constant_time` (42% similarity)
- **Related**: [#8825](https://github.com/llvm/circt/issues/8825) - Switch from hw.inout to custom signal reference type
- **Related**: [#8012](https://github.com/llvm/circt/issues/8012) - Moore to LLVM lowering issues
- **Related**: [#8065](https://github.com/llvm/circt/issues/8065) - Indexing and slicing lowering from Verilog to LLVM IR

## Test Case Validation

| Tool | Status | Notes |
|------|--------|-------|
| Verilator | ✓ Pass | No errors or warnings |
| Icarus Verilog | ✓ Pass | Compiles successfully |
| Slang | ✓ Pass | Build succeeded |
| CIRCT arcilator | ✗ Crash | Assertion failure |

## Suggested Fixes

### Option 1: Add Type Validation (Recommended)
Add proper type checking in `LowerState.cpp:219` before calling `StateType::get()`:
```cpp
if (!isa<llhd::RefType>(arg.getType())) {
  return emitError(arg.getLoc()) << "Unsupported port type for StateType: "
                                 << arg.getType();
}
```

### Option 2: Extend Bit Width Computation
Extend `computeLLVMBitWidth()` to handle `llhd::RefType` by dereferencing the inner type.

### Option 3: Early Rejection
Reject modules with inout ports in the arcilator flow with a meaningful error message before reaching `LowerState`.

## Additional Context

- **Crash Type**: Assertion Failure
- **Severity**: High (Crash/Denial of Service)
- **Reproducibility**: 100% (Minimal 2-line test case)
- **Minimization**: 84.6% reduction from original test case

This is a valid, reproducible crash with a clear root cause. The test case contains only standard IEEE 1800 SystemVerilog features (`inout` port direction) that are accepted by other tools.
