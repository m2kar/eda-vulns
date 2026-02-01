# [Arc] Arcilator crashes with assertion failure on modules with inout ports

## ⚠️ DUPLICATE NOTICE

**This is a duplicate of Issue #9574**: https://github.com/llvm/circt/issues/9574

However, this report provides a **more minimal test case** and additional analysis.

---

## Description

Arcilator crashes with an assertion failure when processing any SystemVerilog module containing `inout` ports. The crash occurs in the `LowerStatePass` when attempting to create state storage for module inputs of type `!llhd.ref<T>`.

**Note**: The issue occurs with ANY inout port, not just when used in sequential logic (as suggested in #9574).

## Minimal Reproducer

Save as `bug.sv`:

```systemverilog
module Bug(inout logic c);
endmodule
```

Run:
```bash
circt-verilog --ir-hw bug.sv | arcilator
```

## Expected Behavior

Either:
1. Successful simulation, or
2. A user-friendly error message such as:
   ```
   error: arcilator does not support inout ports; 'c' has type '!llhd.ref<i1>'
   ```

## Actual Behavior

Assertion failure and crash:

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: .../mlir/include/mlir/IR/StorageUniquerSupport.h:180: Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

**Exit code**: 134 (SIGABRT)

## Stack Trace

```
 #0  llvm::sys::PrintStackTrace(...)
 #1  llvm::sys::RunSignalHandlers()
 #2  SignalHandler(...)
 #3  (/lib/x86_64-linux-gnu/libc.so.6+0x45330)
 ...
#11  (arcilator+0x7dd5bbc)
#12  circt::arc::StateType::get(mlir::Type) .../ArcTypes.cpp.inc:108:3
#13  (anonymous namespace)::ModuleLowering::run() .../LowerState.cpp:219:66
#14  (anonymous namespace)::LowerStatePass::runOnOperation() .../LowerState.cpp:1198:41
 ...
```

## Root Cause Analysis

### Problem Location

`lib/Dialect/Arc/Transforms/LowerState.cpp:215-221`:

```cpp
// Allocate storage for the inputs.
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(arg.getType()), name, storageArg);  // Crash here!
  allocatedInputs.push_back(state);
}
```

### Why It Crashes

1. `circt-verilog --ir-hw` translates `inout` ports to `!llhd.ref<T>` type in HW IR
2. `LowerStatePass` iterates ALL module inputs and calls `StateType::get(arg.getType())`
3. `StateType::verify()` (in `ArcTypes.cpp:80-87`) requires types with computable bit widths
4. `computeLLVMBitWidth()` only handles: `IntegerType`, `hw::ArrayType`, `hw::StructType`, `seq::ClockType`
5. `!llhd.ref<T>` is not handled → returns `nullopt` → assertion fails

### IR Produced

```mlir
module {
  hw.module @Bug(in %c : !llhd.ref<i1>) {
    hw.output
  }
}
```

## Suggested Fixes

### Option 1: Early Validation (Preferred)

Add type validation before attempting state allocation:

```cpp
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  if (isa<llhd::RefType>(arg.getType())) {
    return moduleOp.emitError() 
        << "arcilator does not support inout/ref ports; '"
        << moduleOp.getArgName(arg.getArgNumber())
        << "' has type " << arg.getType();
  }
  // ... rest of allocation
}
```

### Option 2: Handle RefType in computeLLVMBitWidth

Extend `computeLLVMBitWidth()` to handle `llhd::RefType` by computing the bit width of the underlying type:

```cpp
if (auto refType = dyn_cast<llhd::RefType>(type))
  return computeLLVMBitWidth(refType.getNestedType());
```

## Cross-Tool Validation

The test case is valid SystemVerilog:

| Tool | Version | Result |
|------|---------|--------|
| Verilator | 5.022 | ✅ Pass |
| Slang | 10.0.6 | ✅ Pass (0 errors) |

## Environment

- **CIRCT Version**: 1.139.0 (690366b6c)
- **LLVM Version**: 22.0.0git (with assertions)
- **Affected Tool**: arcilator
- **Affected Pass**: LowerStatePass (`arc-lower-state`)
- **Testcase ID**: 260129-00001472

## Additional Notes

The crash only occurs with **assertions-enabled builds**. Non-assertion builds may exhibit undefined behavior or produce incorrect results instead of crashing.

---

## Files Generated

- `bug.sv` - Minimal test case (2 lines)
- `error.log` - Crash output
- `command.txt` - Reproduction command
- `root_cause.md` - Detailed root cause analysis
- `analysis.json` - Structured analysis data
- `validation.json` - Cross-tool validation results
- `duplicates.json` - Duplicate check results

---

**Status**: DUPLICATE of #9574 (do not submit as new issue)

**Recommendation**: Add a comment to #9574 with the more minimal test case showing that sequential logic is not required to trigger the crash.
