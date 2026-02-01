# Validation Report: 260128-000007e8

## Summary
The minimized test case is a single-module SystemVerilog file with an `output string` port. It consistently crashes `circt-verilog --ir-hw` during Moore-to-Core conversion. This should be reported as a CIRCT bug.

## Syntax Check
- **bug.sv**: Valid SystemVerilog syntax.
- **slang**: `Build succeeded: 0 errors, 0 warnings`
- **verilator**: `verilator -sv --lint-only bug.sv` completed without errors.

## Reproduction
Command:
```
export PATH=/opt/firtool/bin:$PATH && circt-verilog --ir-hw bug.sv
```

Result:
- Crash reproduces with a stack dump originating in Moore-to-Core conversion (`SVModuleOpConversion::matchAndRewrite`), consistent with the original crash location (`ModulePortInfo::sanitizeInOut()`).

## Classification
- **Result**: report
- **Reason**: An unsupported `output string` port in SystemVerilog causes a crash rather than a diagnostic.
- **Dialect**: Moore/SystemVerilog
- **Confidence**: high
