# Minimization Report

## Summary

| Metric | Value |
|--------|-------|
| **Original file** | source.sv (14 lines) |
| **Minimized file** | bug.sv (2 lines) |
| **Reduction** | 85.7% |
| **Crash preserved** | Yes |

## Preservation Analysis

### Key Constructs Preserved

Based on `analysis.json`, the following constructs were kept:
- `inout` port declaration (core trigger of the bug)

### Removed Elements

| Element | Lines Removed | Reason |
|---------|---------------|--------|
| `input signed [5:0] a` | 1 | Not needed to trigger bug |
| `input signed [5:0] b` | 1 | Not needed to trigger bug |
| `output signed [5:0] c` | 1 | Not needed to trigger bug |
| `signed [5:0]` width | - | Simplified to 1-bit `inout d` |
| `assign c = a >>> b` | 1 | Arithmetic shift not related to bug |
| `assign d = (b[0]) ? c : a` | 1 | Conditional assignment not needed |
| Comments | 2 | Documentation only |
| Empty lines | 2 | Whitespace only |

### Minimization Rationale

The bug is triggered by the `inout` port declaration, which causes arcilator's `LowerState` pass to fail when processing `!llhd.ref` type. All other constructs (signed types, multiple ports, assignments, arithmetic operations) are not required to reproduce the issue.

## Verification

### Original Assertion
```
state type must have a known bit width; got '!llhd.ref<i6>'
```

### Final Error (Current CIRCT version)
```
error: failed to legalize operation 'arc.state_write'
... (!arc.state<!llhd.ref<i1>>, !llhd.ref<i1>) -> ()
```

**Match**: ✅ Same root cause (`!llhd.ref` type not supported in arcilator)

**Note**: Error message improved in current CIRCT version (cleaner error vs assertion crash), but the underlying bug remains.

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv | arcilator --observe-ports
```

## Minimized Test Case

```systemverilog
module M(inout d);
endmodule
```

## Notes

1. The original test case had 14 lines; minimized to 2 lines (85.7% reduction)
2. Width reduced from `[5:0]` (6-bit) to implicit 1-bit
3. Removed `signed` qualifier as it's not relevant to the bug
4. The bug manifests when arcilator attempts to lower `inout` ports (converted to `!llhd.ref` type in HW dialect)
5. The `--observe-ports` flag is needed to trigger the state lowering path
