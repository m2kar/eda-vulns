# [Arcilator] Assertion failure: StateType does not support LLHD reference types (!llhd.ref<i1>)

## Bug Description

Arcilator crashes with an assertion failure when processing a module containing an `inout` port. The `!llhd.ref<T>` type used to represent bidirectional ports is not supported by `arc::StateType`, which requires types with a known bit width that `computeLLVMBitWidth()` can calculate.

### Error Message
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

## Minimal Test Case

```systemverilog
module M(inout logic x);
endmodule
```

## Steps to Reproduce

```bash
/home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw bug.sv | /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/arcilator
```

### Expected Behavior
Either:
1. Arcilator successfully handles `inout` ports, or
2. Arcilator rejects unsupported constructs with a clear diagnostic error message

### Actual Behavior
Arcilator crashes with an assertion failure.

## Stack Trace

```
#12 circt::arc::StateType::get(mlir::Type) ArcTypes.cpp.inc:108
#13 (anonymous namespace)::ModuleLowering::run() LowerState.cpp:219
#14 (anonymous namespace)::LowerStatePass::runOnOperation() LowerState.cpp:1198
```

Full stack trace:
```
arcilator: /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/IR/StorageUniquerSupport.h:180: static ConcreteT mlir::detail::StorageUserBase<circt::arc::StateType, mlir::Type, circt::arc::detail::StateTypeStorage, mlir::detail::TypeUniquer>::get(MLIRContext *, Args &&...) [ConcreteT = circt::arc::StateType, BaseT = mlir::Type, StorageT = circt::arc::detail::StateTypeStorage, UniquerT = mlir::detail::TypeUniquer, Traits = <>, Args = <mlir::Type &>]: Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
Stack dump:
0.	Program arguments: /edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/arcilator
 #0 0x000056511d5bf23f llvm::sys::PrintStackTrace(llvm::raw_ostream&, int)
 #1 0x000056511d5c0379 llvm::sys::RunSignalHandlers()
 #2 0x000056511d5c0379 SignalHandler(int, siginfo_t*, void*)
 #12 0x000056511dd10bbc circt::arc::StateType::get(mlir::Type)
 #13 0x000056511dd7bf5c (anonymous namespace)::ModuleLowering::run()
 #14 0x000056511dd7bf5c (anonymous namespace)::LowerStatePass::runOnOperation()
 #15 0x0000565121712782 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int)::$_3::operator()() const
```

## Root Cause Analysis

### Crash Location
- **File**: `lib/Dialect/Arc/Transforms/LowerState.cpp`
- **Function**: `ModuleLowering::run()`
- **Line**: 219

### Mechanism
1. SystemVerilog `inout` port is converted to `!llhd.ref<T>` type in the IR
2. `LowerStatePass` processes all module inputs
3. For each input, it creates `RootInputOp` with `StateType::get(arg.getType())`
4. `StateType::get()` calls `verify()` which checks `computeLLVMBitWidth()`
5. `!llhd.ref<i1>` returns `std::nullopt` (unsupported type)
6. Verification fails → Assertion triggered

### Type Verification Code
```cpp
// ArcTypes.cpp:75-81
LogicalResult
StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  Type innerType) {
  if (!computeLLVMBitWidth(innerType))
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

The `computeLLVMBitWidth()` function only supports:
- `seq::ClockType`
- `IntegerType`
- `hw::ArrayType`
- `hw::StructType`

It does NOT support `llhd::RefType` or other LLHD types.

## Suggested Fix Directions

1. **Add support for `llhd::RefType` in `computeLLVMBitWidth()`**: Compute bit width of the inner type for reference types

2. **Add early validation in LowerState**: Check port types before attempting to create `StateType` and emit a proper diagnostic for unsupported types

3. **Filter out inout ports in earlier passes**: If arcilator cannot simulate bidirectional ports, they should be rejected or stripped earlier in the pipeline with a clear error message

4. **Document limitation**: If inout ports are intentionally unsupported, document this limitation clearly

## Cross-Tool Validation

| Tool | Result | Notes |
|------|--------|-------|
| Slang | ✅ pass | Valid IEEE 1800 SystemVerilog syntax |
| Verilator | ✅ pass | Valid syntax |
| Icarus Verilog | ✅ pass | Valid syntax |

The test case uses legal IEEE 1800 SystemVerilog syntax that other tools accept.

## Related Issues

Potentially related but distinct issues:
- #8825: `[LLHD] Switch from hw.inout to a custom signal reference type` - Discusses `llhd.ref<T>` types but from a different angle
- #4916: `[Arc] LowerState: nested arc.state get pulled in wrong clock tree` - Arc and LowerState but focuses on clock tree issues

This issue is unique because it specifically addresses the assertion failure when `StateType` encounters `llhd.ref<T>` types.

## Environment

- **CIRCT Version**: 1.139.0
- **Tool**: arcilator (pipeline: circt-verilog --ir-hw | arcilator)
- **Testcase ID**: 260128-000007ba
