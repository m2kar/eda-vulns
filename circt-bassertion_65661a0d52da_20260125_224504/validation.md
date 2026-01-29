# Validation Report

## Summary
| Item | Result |
|------|--------|
| **Test File** | `bug.sv` |
| **Classification** | `report` (genuine bug) |
| **Syntax Valid** | ✅ Yes (confirmed by slang, verilator) |
| **Cross-Tool Consistent** | ✅ Yes |

## Syntax Validation

### slang (v10.0.6)
```
Build succeeded: 0 errors, 0 warnings
Exit code: 0
```
**Result:** ✅ Valid SystemVerilog

### Verilator (v5.022)
```
Exit code: 0
```
**Result:** ✅ Valid SystemVerilog

### Icarus Verilog (v13.0)
```
bug.sv:1: sorry: Port `str_out` of module `Mod` with type `string` is not supported.
1 error(s) during elaboration.
Exit code: 1
```
**Result:** ⚠️ Unsupported feature (tool limitation, not syntax error)

## IEEE Compliance Analysis

| Aspect | Status |
|--------|--------|
| Standard | IEEE 1800-2017 |
| Feature | `string` type as module port |
| Reference | Section 6.16 (String data type) |
| Syntax | ✅ Valid |

The `string` data type is a valid SystemVerilog type defined in IEEE 1800-2017. Using it as a module port is syntactically valid, though some tools may not support synthesis of string ports.

## CIRCT Behavior Analysis

| Aspect | Value |
|--------|-------|
| Tool | circt-verilog |
| Version | firtool-1.139.0 |
| Exit Code | 139 (SIGSEGV) |
| Crash Type | Segmentation fault |

### Expected Behavior
If `string` type ports are not supported for hardware synthesis, circt-verilog should:
1. Emit a clear error message (e.g., "string type not supported as module port")
2. Exit with a non-zero exit code (but not crash)

### Actual Behavior
circt-verilog crashes with assertion failure:
```
dyn_cast on a non-existent value
```

This is a **bug** because:
1. The compiler crashes instead of reporting an error
2. No useful diagnostic message is provided to the user
3. Stack trace is exposed instead of graceful error handling

## Classification Rationale

| Criterion | Assessment |
|-----------|------------|
| Valid Syntax | ✅ Yes (slang, verilator confirm) |
| Compiler Crash | ✅ Yes (segfault) |
| Graceful Error | ❌ No (assertion failure) |
| Actionable Report | ✅ Yes |

**Conclusion:** This is a **genuine bug** that should be reported to CIRCT.

## Recommendation

**Action:** `report`

The test case demonstrates that:
1. Valid SystemVerilog code causes circt-verilog to crash
2. The crash is due to missing null-check in type conversion (per root cause analysis)
3. Fix should either support the feature or emit proper diagnostic

## Test Case
```systemverilog
module Mod(output string str_out);
endmodule
```

## Reproduction Command
```bash
circt-verilog --ir-hw bug.sv
```
