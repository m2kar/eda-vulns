# Minimization Report

## Summary
Successfully minimized the test case from 500 bytes to 150 bytes (70% reduction).

## Original Test Case (source.sv)
- 27 lines, 500 bytes
- Array declaration with 4 elements
- Struct with `valid` and `data[7:0]` fields
- Verbose with comments

## Minimized Test Case (bug.sv)
- 7 lines, 150 bytes
- No array needed (constant assignment suffices)
- Struct with 2 single-bit fields `a` and `b`
- All comments removed, minimal identifier names

## Removals Made

### 1. Array Eliminated
- **Original**: `logic [7:0] arr [0:3];` with `pkt.data = arr[0];`
- **Minimized**: `x.a = 0;` (constant assignment)
- **Reason**: The bug is triggered by partial struct field assignment, not array indexing

### 2. Struct Simplified
- **Original**: `struct packed { logic valid; logic [7:0] data; }`
- **Minimized**: `struct packed { logic a, b; }`
- **Reason**: Minimal 2-field struct sufficient to trigger the bug

### 3. Comments Removed
- All comments removed as they don't affect behavior

### 4. Identifiers Shortened
- `test` → `m`, `clk` → `clk`, `pkt` → `x`, `q` → `y`, `valid` → `a`, `data` → `b`

## Core Bug Pattern Preserved
```systemverilog
always_comb x.a = 0;       // Partial struct field assignment
always_ff @(posedge clk) y <= x.b;  // Read different field sequentially
```

This pattern triggers the llhd-sig2reg pass to create circular SSA definitions, causing the canonicalize pass to hang.

## Reproduction Command
```bash
timeout 30s circt-verilog --ir-hw bug.sv
```

Expected result: Exit code 124 (timeout)
