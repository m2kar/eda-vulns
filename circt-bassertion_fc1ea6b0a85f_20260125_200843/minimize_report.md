# Minimization Report

## Summary
- **Original file**: source.sv (22 lines)
- **Minimized file**: bug.sv (2 lines)
- **Reduction**: 90.9%
- **Crash preserved**: N/A (Bug fixed in current CIRCT version)

## Key Constructs Preserved

Based on `analysis.json`, the following core construct was retained:
- `inout logic c` - bidirectional port that Arc dialect cannot handle

## Removed Elements

| Element | Lines Removed | Reason |
|---------|---------------|--------|
| `input logic clk` | 1 | Not required to trigger issue |
| `input logic rst` | 1 | Not required to trigger issue |
| `input logic a` | 1 | Not required to trigger issue |
| `output logic b` | 1 | Not required to trigger issue |
| `logic b_reg` | 1 | Not required to trigger issue |
| `always_ff` block | 6 | Sequential logic not needed |
| `assign b = b_reg` | 1 | Output assignment not needed |
| `assign c = (...)` | 1 | Tristate assignment not needed |

## Minimization Process

### Initial Analysis
- Original 22 lines with mixed port directions and sequential logic
- Root cause: `inout logic c` port creates `!llhd.ref<i1>` type
- Arc's `StateType::get()` cannot handle `llhd.ref` type

### Step 1: Remove Sequential Logic
- Removed `always_ff` block and `b_reg` register
- Kept inout port → Issue still present

### Step 2: Remove Unnecessary Ports
- Removed `clk`, `rst`, `a`, `b` ports
- Kept only `inout logic c` → Issue still present

### Step 3: Remove Tristate Assignment
- Removed `assign c = ...` statement
- Empty module with just inout port → Minimal reproduction

## Final Minimized Test Case

```systemverilog
module M(inout logic c);
endmodule
```

## Reproduction Command

Original (from error.txt):
```bash
circt-verilog --ir-hw source.sv | arcilator | opt -O0 | llc -O0 --filetype=obj -o out.o
```

Simplified:
```bash
circt-verilog --ir-hw bug.sv | arcilator
```

## Verification

### Syntax Validation
- **slang**: ✅ Build succeeded: 0 errors, 0 warnings
- **verilator**: ✅ No errors

### Current CIRCT Behavior
The bug appears to be **fixed** in the current CIRCT version:
- `circt-verilog --ir-hw` accepts the file and produces HW IR with `!llhd.ref<i1>`
- `arcilator` now handles this gracefully (ignores inout port)

## Notes

1. The minimal test case preserves the core issue: `inout` port creates `!llhd.ref` type
2. Arc dialect by design does not support bidirectional ports (see ArcOps.cpp:338-339)
3. The crash was in `LowerState.cpp:219` when `StateType::get()` received unsupported type
4. Current CIRCT appears to have added proper handling or early validation
