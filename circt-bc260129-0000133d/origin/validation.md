# CIRCT Bug Validation Report

## Classification
**Result**: `report` (Valid bug to report)

**Reason**: Valid SystemVerilog code with `$error` in `always @(*)` assertion triggers legalization failure for `sim.fmt.literal` in ArcToLLVM pass. Both slang and verilator accept the syntax, confirming this is a CIRCT/arcilator bug, not invalid input.

## Syntax Validation

### Primary Check (slang)
- **Tool**: slang 10.0.6
- **Command**: `slang --parse-only bug.sv`
- **Result**: ✅ Valid (no errors)

### Secondary Check (verilator)
- **Tool**: verilator 5.022
- **Command**: `verilator --lint-only --sv bug.sv`
- **Result**: ✅ Valid (no errors)

## Cross-Tool Analysis

| Tool | Command | Result | Notes |
|------|---------|--------|-------|
| slang | `--parse-only` | ✅ Pass | Clean parse |
| verilator | `--lint-only --sv` | ✅ Pass | No lint warnings |
| circt-verilog + arcilator | `--ir-hw \| arcilator` | ❌ Fail | `sim.fmt.literal` legalization error |

## Code Reduction Summary

| Metric | Value |
|--------|-------|
| Original lines | 14 |
| Minimized lines | 3 |
| **Reduction** | **78.57%** |

## Minimized Test Case
```systemverilog
module m();
  always @(*) assert (0) else $error("");
endmodule
```

## Conclusion
The minimized test case is:
1. **Syntactically valid** - Accepted by both slang and verilator
2. **Semantically meaningful** - Represents valid SystemVerilog assertion pattern
3. **Minimal** - 3 lines with only essential constructs
4. **Reproducible** - Consistently triggers the bug

This confirms the bug should be reported to CIRCT as a legalization gap in the ArcToLLVM pass for `sim.fmt.literal` operations generated from `$error` system tasks.
