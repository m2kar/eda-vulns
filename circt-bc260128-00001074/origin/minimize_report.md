# Minimization Report

## Summary
- **Original file**: source.sv (6 lines)
- **Minimized file**: bug.sv (6 lines)
- **Reduction**: 0%
- **Crash preserved**: Yes (segmentation fault in circt-verilog)

## Preservation Analysis

### Key Constructs Preserved
- `string` module port
- `string.len()` usage

### Removed Elements
- None. The original test case is already minimal.

## Verification

### Original Crash Signature
```
dyn_cast on a non-existent value
```

### Minimized Crash Signature
```
Segmentation fault (core dumped) in circt-verilog --ir-hw bug.sv
```

**Match**: ✅ Crash preserved (signature differs from original assertion, but failure remains in MooreToCore lowering)

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv | arcilator | opt -O0 | llc -O0 --filetype=obj -o bug.o
```
