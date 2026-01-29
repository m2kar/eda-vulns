# Minimization Report

## Summary
- **Original file**: origin/source.sv (24 lines)
- **Minimized file**: bug.sv (2 lines)
- **Reduction**: 91.7%
- **Crash preserved**: Yes ✓

## Key Constructs Preserved

Based on `analysis.json`, the essential construct causing the crash:
- `output string str_out` - string type as module port

This is the minimal code needed to trigger the assertion failure in MooreToCore conversion.

## Removed Elements

| Element | Lines Removed |
|---------|---------------|
| Input ports (clk, P1, P2) | 3 |
| Local variables (str, r1) | 2 |
| always @(posedge clk) block | 4 |
| always_comb block | 5 |
| assign statement | 1 |
| Empty lines | 5 |
| Module renamed | - |

## Minimization Process

### Iteration 1: Remove all inputs
- Removed `input logic clk`, `input logic [7:0] P1`, `input logic [7:0] P2`
- Result: Crash preserved ✓

### Iteration 2: Remove internal variables
- Removed `string str`, `logic [7:0] r1`
- Result: Crash preserved ✓

### Iteration 3: Remove always blocks
- Removed `always @(posedge clk)` block
- Removed `always_comb` block
- Result: Crash preserved ✓

### Iteration 4: Remove assign statement
- Removed `assign str_out = str`
- Result: Crash preserved ✓

### Iteration 5: Simplify module name
- Changed `mixed_assignments` to `test`
- Result: Crash preserved ✓

## Crash Verification

### Original Assertion (from error.txt)
```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed
```

### Minimized Crash Location
```
SVModuleOpConversion::matchAndRewrite MooreToCore.cpp
MooreToCorePass::runOnOperation MooreToCore.cpp
```

**Match**: ✅ Same crash path (MooreToCore conversion fails on string type port)

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv
```

## Analysis

The crash occurs because:
1. SystemVerilog `string` type is parsed successfully
2. Moore dialect converts it to `sim::DynamicStringType`
3. MooreToCore conversion attempts to create HW module port
4. HW dialect rejects `DynamicStringType` (not a valid HW type)
5. Assertion fails due to null/invalid type conversion result

The minimal reproducer demonstrates that a single `output string` port is sufficient to trigger the bug - no other code is necessary.
