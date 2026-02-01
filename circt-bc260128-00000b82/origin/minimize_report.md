# Minimization Report

## Summary
- Root cause pattern preserved: `output string s[1:0]` (unpacked array of strings)
- Removed inputs, clocking, and procedural blocks; only module port remains

## Minimization Steps
1. Kept the module port declaration that triggers Moore-to-Core conversion.
2. Removed clock/input ports and procedural blocks unrelated to port type lowering.
3. Re-ran `circt-verilog --ir-hw bug.sv` to confirm the assertion persists.

## Reduction
- Original lines: 19 (source.sv)
- Minimized lines: 4 (bug.sv)
- Reduction: 78.95%

## Reproduction
```
/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw bug.sv
```
