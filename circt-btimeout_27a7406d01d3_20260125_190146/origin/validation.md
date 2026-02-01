# Validation Report

## Summary
| Field | Value |
|-------|-------|
| **Classification** | `report` (Genuine Bug) |
| **Confidence** | High |
| **Test Case** | bug.sv (6 lines) |
| **Reduction** | 82.4% from original |

## Syntax Validation

### Slang (Reference SystemVerilog Compiler)
```
Build succeeded: 0 errors, 0 warnings
```
**Status**: ✅ PASS

### Verilator
```
%Warning-DECLFILENAME: Filename does not match MODULE name
%Warning-UNUSEDSIGNAL: Bits of signal not used: s[0]
%Warning-UNDRIVEN: Bits of signal not driven: s[1]
```
**Status**: ⚠️ Warnings only (no errors)

**Analysis**: 
- DECLFILENAME: Cosmetic warning about filename/module name mismatch
- UNUSEDSIGNAL/UNDRIVEN: These warnings are **expected** - they correctly identify the partial struct assignment pattern that triggers the CIRCT bug

### Icarus Verilog
```
always_comb process has no sensitivities
```
**Status**: ✅ PASS (warning is acceptable for constant assignment)

## Cross-Tool Validation Results

| Tool | Syntax | Compilation | Notes |
|------|--------|-------------|-------|
| Slang | ✅ Pass | ✅ Success | Reference SV compiler |
| Verilator | ✅ Pass | ✅ Success | Lint warnings as expected |
| Icarus Verilog | ✅ Pass | ✅ Success | Minor warning |
| CIRCT (arcilator) | ✅ Pass | ❌ **TIMEOUT** | Bug triggered |

## Bug Classification

### Is this a genuine bug?
**YES** - The test case is valid SystemVerilog that compiles successfully with multiple industry-standard tools (Slang, Verilator, Icarus Verilog) but causes CIRCT's arcilator pipeline to enter an infinite loop.

### Is this a design limitation?
**NO** - Partial struct field assignment is a common and valid SystemVerilog pattern. The bug is in how CIRCT handles the conversion to hw.struct_inject operations.

### Is this an invalid test case?
**NO** - The syntax is valid per IEEE 1800-2017 SystemVerilog standard. Slang (the reference SV compiler) accepts it without any warnings.

### Is this a feature request?
**NO** - This is a bug in existing functionality, not a request for new features. CIRCT claims to support SystemVerilog and should handle this valid code pattern.

## Minimal Reproducer

```systemverilog
module M(output logic O);
  typedef struct packed { logic a; logic b; } S;
  S s;
  always_comb s.b = 0;
  assign O = s.a;
endmodule
```

### Reproduction Command
```bash
/opt/firtool/bin/circt-verilog --ir-hw bug.sv | /opt/firtool/bin/arcilator | /opt/firtool/bin/opt -O0 | /opt/firtool/bin/llc -O0 --filetype=obj
```

**Expected**: Compilation succeeds
**Actual**: Timeout after 60 seconds (exit code 124)

## Root Cause
Partial assignment to packed struct field in `always_comb` creates a false combinational loop when converted to `hw.struct_inject` operations. The arcilator LowerState pass enters an infinite loop processing the cyclic dependency.

## Recommendation
This bug should be reported to the CIRCT project. The test case demonstrates a clear regression where valid SystemVerilog code causes compilation to hang indefinitely.

## Verdict
**Classification: `report`** - This is a genuine bug that should be reported to the CIRCT issue tracker.
