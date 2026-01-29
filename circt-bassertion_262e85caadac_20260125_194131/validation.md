# Test Case Validation Report

## Reduction Summary
- **Original**: 19 lines
- **Minimized**: 4 lines
- **Reduction**: 78.9%

## Minimized Test Case
```systemverilog
module m(input c, output logic q);
  always @(negedge c) q <= 0;
  assign q = q;
endmodule
```

## Reproduction Command
```bash
/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw bug.sv
```

## Syntax Check

**Result**: Semantically invalid

The test case violates IEEE 1800 SystemVerilog rules:
1. **Mixed assignments**: Continuous (`assign`) and procedural (`<=`) assignments to the same variable `q`
2. **Self-referential**: `assign q = q` creates circular logic

However, this should result in a **diagnostic error**, not a compiler crash.

## Cross-Tool Validation

### Verilator (v5.022)
**Result**: Error (as expected)

```
%Error: bug.sv:3:12: Wire inputs its own output, creating circular logic (wire x=x)
%Error-BLKANDNBLK: Unsupported: Blocked and non-blocking assignments to same variable: 'q'
```

Verilator correctly identifies both the circular assignment and the mixed blocking/non-blocking issue.

### Slang (v10.0.6)
**Result**: Error (as expected)

```
bug.sv:3:10: error: cannot mix continuous and procedural assignments to variable 'q'
```

Slang correctly rejects this as a semantic error with a clear diagnostic.

## Classification

**Result: report**

### Reason
This is a **compiler robustness bug**. While the test case is semantically invalid (mixing continuous and procedural assignments is not allowed in SystemVerilog), CIRCT should:

1. **Detect** the invalid construct during semantic analysis
2. **Emit** a clear error diagnostic
3. **Exit gracefully** without crashing

Instead, CIRCT:
- Fails to validate the IR invariants
- Attempts to RAUW (Replace All Uses With) a value with itself during Sig2RegPass
- Crashes with an assertion failure

### Technical Details
- **Crash Location**: `Sig2RegPass.cpp:207` in `SigPromoter::isPromotable()`
- **Assertion**: `cannot RAUW a value with itself`
- **Root Cause**: Self-referential assignment creates a circular value dependency that passes initial validation but violates internal IR invariants during optimization

## Features Used
- `always @(negedge clock)` - edge-triggered procedural block
- `assign` - continuous assignment
- Self-referential assignment (`assign q = q`)
- Multiple drivers to same signal (always + assign)

## Known Limitations
None applicable - this is a bug, not a limitation.

## Recommendations

**Should Report**: ✅ Yes

This bug should be reported because:
1. **Crashes are never acceptable** - compilers must handle all inputs gracefully
2. **Other tools handle it correctly** - Verilator and Slang both produce clear error messages
3. **Security concern** - crashes from malformed input could be exploited
4. **User experience** - assertion failures are confusing; clear diagnostics help users fix their code

### Suggested Fix Direction
Add validation in the Moore dialect or during lowering to detect:
1. Self-referential continuous assignments
2. Mixed continuous/procedural assignments to the same variable

Emit appropriate diagnostics before the Sig2RegPass runs.
