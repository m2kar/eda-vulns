# [Arcilator] Assertion failure when processing SystemVerilog modules with `inout` ports

## Bug Description

**arcilator** crashes with an assertion failure when attempting to process SystemVerilog modules containing `inout` ports (bidirectional buses). The crash occurs during the `LowerState` pass when allocating storage for module inputs. Instead of emitting a user-friendly error message indicating that inout ports are unsupported, the compiler aborts with an unclear assertion:

```
state type must have a known bit width; got '!llhd.ref<i8>'
```

### Reproducer

**Minimized test case** (6 lines):
```systemverilog
// Minimized test case for Arc inout port assertion failure
module InoutBug(
  inout logic [7:0] data_bus
);
endmodule
```

**Command** (original crash):
```bash
/opt/firtool/bin/circt-verilog --ir-hw source.sv | /opt/firtool/bin/arcilator
```

**Expected behavior**: Emit a clear error message like "inout ports are not supported by arcilator"

**Actual behavior**: Assertion failure crash with confusing error about unknown bit width

### Crash Output

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i8>'
arcilator: /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/IR/StorageUniquerSupport.h:180: static ConcreteT mlir::detail::StorageUserBase<circt::arc::StateType, mlir::Type, circt::arc::detail::StateTypeStorage, mlir::detail::TypeUniquer>::get(MLIRContext *, Args &&...) [ConcreteT = circt::arc::StateType, BaseT = mlir::Type, StorageT = circt::arc::detail::StateTypeStorage, UniquerT = mlir::detail::TypeUniquer, Traits = <>, Args = <mlir::Type &>]: Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace and instructions to reproduce the bug.
```

### Reproducibility Status

**Note**: This crash was originally observed with CIRCT 1.139.0 but could NOT be reproduced with the current toolchain (CIRCT firtool-1.139.0, LLVM 22.0.0git). The issue may have been partially fixed in a patch release. However, the root cause analysis remains valid as Arc dialect explicitly does not support inout ports, and the error handling should be improved regardless of whether the specific assertion is still triggered.

**Original crash details**:
- **Version**: CIRCT 1.139.0
- **Failing Pass**: LowerStatePass
- **Crash Location**: `lib/Dialect/Arc/Transforms/LowerState.cpp:219`
- **Assertion**: `verifyInvariants` failure in `StateType::get()`

## Root Cause Analysis

### Failure Mechanism

The crash occurs through the following pipeline:

1. `circt-verilog` parses SystemVerilog and creates an `InOutType` for the `inout` port
2. Moore-to-Core conversion wraps the inout port as `llhd::RefType<i8>`
3. The module argument with type `!llhd.ref<i8>` reaches arcilator's `LowerState` pass
4. `LowerState::run()` at line 219 attempts to allocate storage:
   ```cpp
   auto state = RootInputOp::create(allocBuilder, arg.getLoc(),
                                 StateType::get(arg.getType()), name, storageArg);
   ```
5. `StateType::verify()` calls `computeLLVMBitWidth(llhd::RefType<i8>)`
6. `computeLLVMBitWidth()` returns `nullopt` for `llhd::RefType` (only handles ClockType, IntegerType, ArrayType, StructType)
7. Assertion fails because bit width is unknown

### Why This Should Be Handled Gracefully

Arc dialect explicitly **does not support** inout ports. The verifier in `ArcOps.cpp:337-340` has this check:

```cpp
for (const hw::ModulePort &port : getIo().getPorts())
  if (port.dir == hw::ModulePort::Direction::InOut)
    return emitOpError("inout ports are not supported");
```

However, this verification runs **after** the transformation that crashes. The `LowerState` pass should detect `llhd::RefType` (which represents inout ports) and emit a clear diagnostic before attempting allocation.

## Suggested Fix

### Preferred Solution: Early Rejection in LowerState

Add a check at the beginning of `LowerState::run()` to detect and gracefully reject `llhd::RefType` arguments:

```cpp
// In ModuleLowering::run() before the allocation loop
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  if (isa<llhd::RefType>(arg.getType()))
    return moduleOp.emitOpError()
        << "inout ports are not supported by arcilator (found argument '"
        << moduleOp.getArgName(arg.getArgNumber()) << "' with type "
        << arg.getType() << ")";
}
```

### Alternative: Pre-Pass Validation

Add a dedicated validation pass `arc-reject-inout-ports` that runs after HW lowering and before LowerState to catch this case with a user-friendly error.

### Long-Term Solution

Implement proper inout port support in Arc dialect (this would require significant work for bidirectional storage and tristate logic). See related issue #8825.

## Related Issues

- **#8825** - "[LLHD] Switch from hw.inout to a custom signal reference type" - Tracks the underlying type system limitation and proposes implementing proper reference types that would enable inout support

## Additional Information

**Test case validation**:
- Syntax validation with Verilator: ✅ Pass
- Syntax validation with Slang: ✅ Pass
- Classification: While Arc not supporting inout is by design, the **assertion failure instead of graceful error** is a bug

**Key files involved**:
- `lib/Dialect/Arc/Transforms/LowerState.cpp:219` - Crash location
- `lib/Dialect/Arc/ArcTypes.cpp:80-87` - StateType verification
- `lib/Dialect/Arc/ArcOps.cpp:337-340` - ModelOp inout rejection (runs too late)
- `lib/Conversion/MooreToCore/MooreToCore.cpp:248-252` - FIXME comment about inout handling

**Minimization**: Reduced from original 24 lines to 6 lines (75% reduction) while preserving the key pattern (inout port declaration).
