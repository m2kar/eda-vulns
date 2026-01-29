# [circt-verilog][arcilator] Assertion failure when processing modules with `inout` ports

## Bug Description

`arcilator` crashes with an assertion failure when processing SystemVerilog modules that contain `inout` (bidirectional/tristate) ports. The crash occurs because `circt-verilog --ir-hw` generates LLHD reference types (`!llhd.ref<T>`) for `inout` ports, but the Arc dialect's `StateType::verify()` function does not handle LLHD reference types, causing a verification failure.

## Minimal Reproducible Example

```systemverilog
module M(inout wire c);
endmodule
```

## Steps to Reproduce

1. Create a file with the minimal test case above (`bug.sv`)
2. Run the following command:
   ```bash
   circt-verilog --ir-hw bug.sv | arcilator
   ```
3. Observe the crash

## Expected Behavior

One of the following:
- Successfully compile the module with `inout` port support
- Emit a clear, user-friendly error message: "arcilator does not support bidirectional (inout) ports"

## Actual Behavior

The tool crashes with an internal assertion failure:

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: ... Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

Stack trace (relevant frames):
```
#12 circt::arc::StateType::get(mlir::Type) ArcTypes.cpp.inc:108
#13 (anonymous namespace)::ModuleLowering::run() LowerState.cpp:219
#15 (anonymous namespace)::LowerStatePass::runOnOperation() LowerState.cpp:1198
```

## Root Cause Analysis

### Failure Path

1. `circt-verilog --ir-hw` parses SystemVerilog and produces MLIR IR
2. For `inout` ports, the Moore dialect generates LLHD reference types (`!llhd.ref<i1>`)
3. When piped to `arcilator`, the Arc dialect's `LowerStatePass` attempts to allocate state storage for module inputs
4. At `LowerState.cpp:219`, it calls `StateType::get(arg.getType())` with the LLHD ref type
5. `StateType::verify()` calls `computeLLVMBitWidth()` to check if the type has a known bit width
6. `computeLLVMBitWidth()` only handles: `ClockType`, `IntegerType`, `ArrayType`, `StructType`
7. LLHD reference types fall through and return `std::nullopt`, causing verification to fail
8. The verification failure triggers the assertion crash

### Code Location

**File**: `lib/Dialect/Arc/ArcTypes.cpp`
**Function**: `computeLLVMBitWidth()`

The function lacks a handler for `llhd::RefType`:
```cpp
static std::optional<uint64_t> computeLLVMBitWidth(Type type) {
  if (isa<seq::ClockType>(type))
    return 1;
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  if (auto arrayType = dyn_cast<hw::ArrayType>(type))
    // ... handles arrays
  if (auto structType = dyn_cast<hw::StructType>(type))
    // ... handles structs
  // We don't know anything about any other types.
  return {};  // <- LLHD ref types fall through here!
}
```

## Test Case Validation

The minimal test case (`bug.sv`) has been validated against multiple SystemVerilog tools:

| Tool | Result | Exit Code |
|------|---------|-----------|
| Verilator (`verilator --lint-only bug.sv`) | ✅ Pass | 0 |
| Slang (`slang bug.sv`) | ✅ Pass | 0 |
| Icarus Verilog (`iverilog -g2005-sv bug.sv`) | ✅ Pass | 0 |

All tools accept this as valid IEEE 1800 SystemVerilog syntax.

## Environment

- **CIRCT Version**: 1.139.0
- **Tool**: arcilator
- **Dialect**: Arc, LLHD (from Moore lowering)
- **Failing Pass**: LowerStatePass (arc-lower-state)
- **Crash Type**: Assertion failure

## Related Issues

This is a **new issue**, but related to ongoing work on LLHD dialect integration:

- **#9467**: `[arcilator] fails to lower llhd.constant_time` - Similar class of bug (arcilator + LLHD incompatibility), but different failing operation
- **#8825**: `[LLHD] Switch from hw.inout to a custom signal reference type` - Design discussion introducing `llhd.ref<T>` type (upstream context)
- **#8845**: `[circt-verilog] produces non comb/seq dialects including cf and llhd` - Related issue about unexpected LLHD dialect in output
- **#9469**: `[arcilator] Inconsistent compilation behavior: array indexing in sensitivity list` - Different trigger, but shows arcilator + LLHD integration challenges (CLOSED)

## Suggested Fixes

1. **Add LLHD Type Support in Arc**: Extend `computeLLVMBitWidth()` to handle `llhd::RefType` by extracting the underlying type
2. **Proper Error Reporting**: Detect unsupported types early in the arcilator pipeline and emit a clear diagnostic message instead of crashing
3. **Moore-to-HW Lowering**: Ensure `--ir-hw` flag properly converts all LLHD types to HW equivalents or rejects unsupported constructs explicitly
4. **Documentation**: If inout support is not intended for arcilator, document this limitation clearly

## Additional Context

- The `--ir-hw` flag suggests that `circt-verilog` should output HW dialect, but LLHD types still leak through for `inout` ports
- `inout` ports are a fundamental HDL feature (tristate/bidirectional logic) used in many real designs
- Even if arcilator doesn't intend to support inout ports, the current behavior (assertion crash exposing implementation details) is a poor user experience
