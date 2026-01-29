# [arcilator] Assertion failure: "state type must have a known bit width" with inout ports

## Description

`arcilator` crashes with an assertion failure when processing a module containing `inout` ports. The `LowerStatePass` attempts to wrap `llhd::RefType` (used to represent `inout` ports) into `arc::StateType`, but `StateType` only supports types with known bit widths and does not handle `RefType`.

This is a valid IEEE 1800-2005 SystemVerilog construct that should either be properly handled or rejected with a meaningful error message instead of causing an assertion failure.

## Minimal Reproducer

```systemverilog
module MixedPorts(inout wire c);
endmodule
```

## Steps to Reproduce

1. Save the above code to `test.sv`
2. Run:
   ```bash
   circt-verilog --ir-hw test.sv 2>&1 | arcilator
   ```
3. Observe the crash

## Error Output

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: .../mlir/include/mlir/IR/StorageUniquerSupport.h:180: Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### Key Stack Frames

| Frame | Function | Location |
|-------|----------|----------|
| #12 | `circt::arc::StateType::get(mlir::Type)` | ArcTypes.cpp.inc:108 |
| #13 | `ModuleLowering::run()` | LowerState.cpp:219 |
| #14 | `LowerStatePass::runOnOperation()` | LowerState.cpp:1198 |

## Root Cause Analysis

### Mechanism

1. SystemVerilog module contains `inout wire c` port
2. `circt-verilog --ir-hw` outputs HW IR where inout ports are represented as `!llhd.ref<i1>` type
3. `arcilator` runs `LowerStatePass` to allocate storage for module inputs
4. `LowerStatePass::ModuleLowering::run()` (line 219) calls `StateType::get(arg.getType())` for each argument
5. `StateType::verify()` calls `computeLLVMBitWidth()` which only supports: `seq::ClockType`, `IntegerType`, `hw::ArrayType`, `hw::StructType`
6. `llhd::RefType` is **not supported** → `computeLLVMBitWidth()` returns `nullopt` → assertion failure

### Key Findings

- `arcilator`'s `StateType` does not support `llhd::RefType` (used for inout/ref ports)
- `computeLLVMBitWidth()` in `ArcTypes.cpp` lacks a handler for `RefType`
- Other passes (e.g., `HWToSystemC`) explicitly check and reject inout ports with meaningful errors
- `LowerStatePass` lacks this early validation

## Validation

### Cross-Tool Verification

| Tool | Status | Version |
|------|--------|---------|
| Verilator | ✅ Pass | 5.022 |
| Slang | ✅ Pass | 10.0.6 |
| Icarus Verilog | ✅ Pass | 13.0 |

All three tools confirm the test case is syntactically valid per IEEE 1800-2005.

## Related Issues

- **#8825** - [LLHD] Switch from hw.inout to a custom signal reference type
  - Discusses the design of `llhd.ref<T>` type that causes this crash
  - May provide context for the fix direction

## Additional Information

- **CIRCT Version**: 1.139.0
- **Crash Location**: `LowerState.cpp:219` in `ModuleLowering::run()`
- **Failing Pass**: `LowerStatePass`
- **Crash Type**: Assertion failure
- **Note**: Bug appears fixed in current `/opt/firtool/bin/` toolchain (no crash)

## Suggested Fix Direction

### Option 1: Early Error Detection (Recommended)

Add validation at `LowerStatePass` entry to detect unsupported port types and emit a meaningful diagnostic:

```cpp
// At the beginning of ModuleLowering::run()
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  if (isa<llhd::RefType>(arg.getType())) {
    return moduleOp.emitError()
        << "arcilator does not support inout/ref ports; port '"
        << moduleOp.getArgName(arg.getArgNumber()) << "' has type "
        << arg.getType();
  }
}
```

### Option 2: Extend StateType Support

Add `llhd::RefType` handling in `computeLLVMBitWidth()`:

```cpp
if (auto refType = dyn_cast<llhd::RefType>(type)) {
  return computeLLVMBitWidth(refType.getNestedType());
}
```

### Option 3: Pipeline-Level Fix

Add a pass before `LowerStatePass` to convert inout ports to equivalent input + output pairs.
