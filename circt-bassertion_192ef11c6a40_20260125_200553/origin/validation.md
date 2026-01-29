# Validation Report

## Summary

| Field | Value |
|-------|-------|
| Test Case | `bug.sv` |
| **Result** | **report** |
| Classification | Genuine Bug |
| Confidence | High |

## Syntax Validation

### IEEE Standard Compliance

The test case is compliant with **IEEE 1800** (SystemVerilog) standard:

```systemverilog
module M(inout wire c);
endmodule
```

- `inout` port direction: Valid since Verilog-1995
- `wire` net type: Valid since Verilog-1995
- Module with single port: Valid syntax

**Verdict: ✅ Syntactically correct**

## Cross-Tool Validation

| Tool | Command | Result | Exit Code |
|------|---------|--------|-----------|
| Verilator | `verilator --lint-only bug.sv` | ✅ Pass | 0 |
| Slang | `slang bug.sv` | ✅ Pass | 0 |
| Icarus Verilog | `iverilog -g2005-sv bug.sv` | ✅ Pass | 0 |

**All three independent tools accept this code as valid SystemVerilog.**

## Bug Classification

### Why This Is a Genuine Bug

1. **Valid Syntax**: The test case uses standard SystemVerilog syntax accepted by all major tools
2. **Assertion Failure**: The tool crashes with an internal assertion, not a user-facing error
3. **Fundamental Feature**: `inout` ports are a basic HDL feature used in real designs
4. **No Graceful Degradation**: Even if arcilator doesn't support inout, it should emit a clear diagnostic message

### Expected Behavior

One of:
- Successfully compile the module with inout port
- Emit a clear error message: "arcilator does not support inout/bidirectional ports"

### Actual Behavior

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: ...Assertion `succeeded(...)' failed.
```

This is an **internal assertion failure** that exposes implementation details to the user.

## Root Cause Summary

The crash occurs because:
1. `circt-verilog --ir-hw` produces `!llhd.ref<i1>` type for inout ports
2. `arcilator`'s LowerState pass calls `StateType::get()` with this type
3. `StateType::verify()` fails because `computeLLVMBitWidth()` doesn't handle LLHD ref types
4. The verification failure triggers an assertion crash

## Recommendation

**File as bug report** to the CIRCT project (https://github.com/llvm/circt)

### Suggested Fix Directions

1. **Add support**: Handle `llhd.ref` types in `computeLLVMBitWidth()`
2. **Better error**: Reject inout ports with a clear error message in arcilator
3. **Moore lowering**: Ensure `--ir-hw` doesn't produce LLHD types when HW dialect is requested

## Conclusion

| Criteria | Assessment |
|----------|------------|
| Valid test case | ✅ Yes |
| Genuine bug | ✅ Yes |
| Reportable | ✅ Yes |
| Severity | High (crash) |

**Final Verdict: REPORT**
