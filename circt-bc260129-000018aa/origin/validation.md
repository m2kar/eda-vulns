# Validation Report

## Summary

| Field | Value |
|-------|-------|
| Testcase ID | 260129-000018aa |
| Validation Result | **report** |
| Bug Type | Infinite loop / exponential complexity |
| Affected Tool | circt-verilog --ir-hw |
| Severity | High |

## Syntax Validation

| Tool | Command | Result | Exit Code |
|------|---------|--------|-----------|
| Verilator | `verilator --lint-only bug.sv` | ✅ Pass | 0 |
| Slang | `slang --lint-only bug.sv` | ✅ Pass | 0 |

**Conclusion:** The testcase is syntactically valid SystemVerilog.

## CIRCT Pipeline Validation

| Stage | Command | Result | Exit Code |
|-------|---------|--------|-----------|
| Parse | `circt-verilog --parse-only bug.sv` | ✅ Pass | 0 |
| Moore IR | `circt-verilog --ir-moore bug.sv` | ✅ Pass | 0 |
| HW IR | `circt-verilog --ir-hw bug.sv` | ❌ **Timeout** | 124 |

**Conclusion:** Bug occurs during MooreToCore conversion (--ir-hw stage).

## Cross-Tool Comparison

| Tool | Behavior |
|------|----------|
| Verilator | ✅ Accepts |
| Slang | ✅ Accepts |
| CIRCT (parse) | ✅ Accepts |
| CIRCT (ir-hw) | ❌ Hangs indefinitely |

**Conclusion:** This is a CIRCT-specific bug. Other tools handle the testcase correctly.

## Minimized Testcase

```systemverilog
// Minimal testcase for circt-verilog --ir-hw timeout
// Bug: Compilation hangs when processing always_comb with bit-select
module top;
  logic [7:0] data;
  always_comb data[0] = ~data[7];
endmodule
```

## Reproduction Steps

```bash
# 1. Save the above code to bug.sv
# 2. Run with timeout:
timeout 30s circt-verilog --ir-hw bug.sv
# Expected: Hangs until timeout (exit code 124)
```

## Key Findings

1. **Simple Trigger:** The bug is triggered by a minimal pattern:
   - `always_comb` block
   - Bit-select assignment: `data[0] = ~data[7]`
   - Signal width ≥ 8 bits

2. **NOT Required:** The original analysis suspected:
   - ❌ Nested module structures
   - ❌ Function call chains
   - ❌ Complex signal dependencies

3. **Related Bug:** With smaller bit widths (< 8 bits), a different bug occurs:
   - Segmentation fault in `circt::comb::XorOp::canonicalize`
   - This is a separate issue that should be reported separately

## Recommendation

**Report this bug to CIRCT project** with:
- Minimized testcase (bug.sv)
- Reproduction command
- Analysis showing it affects MooreToCore conversion

## Files Generated

- `bug.sv` - Minimized testcase
- `error.log` - Error information
- `command.txt` - Reproduction command
- `minimize_report.md` - Minimization process details
- `validation.json` - Structured validation data
- `validation.md` - This report
