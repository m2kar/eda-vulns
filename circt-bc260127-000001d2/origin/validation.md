# Validation Report

## Summary
**Classification**: `report` - This is a genuine bug in CIRCT arcilator

## Syntax Validation

### Testcase (bug.sv)
```systemverilog
module m(input q);
  always_comb assert(q) else $error("");
endmodule
```

### Syntax Check Results
| Tool | Result |
|------|--------|
| Verilator 5.022 | ✅ lint-only passed |
| Slang 10.0.6 | ✅ 0 errors, 0 warnings |
| circt-verilog | ✅ Parses correctly |

**Conclusion**: The testcase is valid SystemVerilog syntax.

## Cross-Tool Verification

### Verilator
```bash
$ verilator --lint-only bug.sv
# No output (success)
```

### Slang
```bash
$ slang bug.sv
Top level design units:
    m
Build succeeded: 0 errors, 0 warnings
```

### CIRCT circt-verilog (parsing only)
```bash
$ circt-verilog --ir-hw bug.sv
```
Generates valid MLIR:
```mlir
module {
  hw.module @m(in %q : i1) {
    %0 = sim.fmt.literal "Error: "
    llhd.combinational {
      cf.cond_br %q, ^bb2, ^bb1
    ^bb1:
      sim.proc.print %0
      cf.br ^bb2
    ^bb2:
      llhd.yield
    }
    hw.output
  }
}
```

Note: The `sim.fmt.literal` is correctly linked to `sim.proc.print` in the IR.

## Bug Analysis

### Failure Point
- **Tool**: arcilator
- **Pass**: LowerArcToLLVM
- **Operation**: `sim.fmt.literal`

### Error Message
```
<stdin>:3:10: error: failed to legalize operation 'sim.fmt.literal'
    %0 = sim.fmt.literal "Error: "
```

### Root Cause
The LowerArcToLLVM pass in arcilator marks `sim::FormatLiteralOp` as legal (expecting it to be removed by DCE as a Pure operation), but the operation is not properly consumed during conversion. The `sim.fmt.literal` is correctly used by `sim.proc.print` in the MLIR, but the legalization pattern fails to handle this case.

## Classification Rationale

| Criterion | Assessment |
|-----------|------------|
| Valid syntax | ✅ Yes - accepted by Verilator and Slang |
| Standard construct | ✅ Yes - immediate assertion with $error() is IEEE 1800-2017 compliant |
| Correct IR generation | ✅ Yes - circt-verilog generates proper MLIR |
| Unexpected failure | ✅ Yes - legalization fails unexpectedly |
| Not a documented limitation | ✅ Yes - assertions should be supported |

**Result**: This is a **genuine bug** that should be reported.

## Recommendation

Submit bug report to CIRCT GitHub repository with:
1. Minimal testcase (3 lines)
2. Reproduction command
3. Error output showing legalization failure
4. Analysis of root cause in LowerArcToLLVM pass
