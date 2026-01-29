# Validation Report

## Test File Summary

| Property | Value |
|----------|-------|
| File | `bug.sv` |
| Lines | 2 |
| Syntax Valid | ✅ Yes |
| Classification | **report** (genuine bug) |
| Confidence | High |

## Test Case

```systemverilog
module MixedPorts(inout wire c);
endmodule
```

## Syntax Analysis

The test case uses standard IEEE 1800-2005 SystemVerilog syntax:

- **`inout wire`**: Valid bidirectional port declaration per IEEE 1800-2005 Section 23.2.2 "Port declarations"
- **Module structure**: Minimal valid module with a single port

## Cross-Tool Validation

All three validation tools confirm the syntax is correct:

| Tool | Version | Status | Notes |
|------|---------|--------|-------|
| Verilator | 5.022 | ✅ Pass | No warnings or errors |
| Slang | 10.0.6 | ✅ Pass | "Build succeeded: 0 errors, 0 warnings" |
| Icarus Verilog | 13.0 | ✅ Pass | No warnings or errors |

### Verilator

```bash
$ verilator --lint-only bug.sv
# (no output - clean)
```

### Slang

```bash
$ slang bug.sv
Top level design units:
    MixedPorts

Build succeeded: 0 errors, 0 warnings
```

### Icarus Verilog

```bash
$ iverilog -Wall bug.sv -o /dev/null
# (no output - clean)
```

## Classification

### Result: **report** (Genuine Bug)

**Reason**: The test case is syntactically valid according to the IEEE 1800-2005 standard and passes validation by three independent SystemVerilog tools. The CIRCT arcilator crashes with an assertion failure when processing this valid input.

### Evidence

1. **Valid Syntax**: All three cross-tool validators accept the input without errors
2. **Standard Construct**: `inout wire` is a basic SystemVerilog port type
3. **Compiler Crash**: arcilator aborts with assertion failure instead of graceful error handling
4. **Root Cause**: LowerStatePass attempts to create `StateType` for `llhd::RefType`, which is unsupported

## Recommendation

### Action: **submit_issue**

**Reason**: This is a genuine compiler bug that should be reported to the CIRCT project:

1. Valid SystemVerilog input causes assertion failure
2. The crash location is well-identified (`LowerState.cpp:219`)
3. The root cause is understood (missing support for `llhd::RefType` in `StateType`)
4. A minimal reproducible test case is available

### Suggested Fix Direction

Either:
1. Add early detection to reject `inout` ports with a meaningful error message
2. Implement proper handling for `llhd::RefType` in the LowerStatePass

## Validation Metadata

- **Timestamp**: 2026-01-28T11:20:18+00:00
- **Validation Version**: 1.0
