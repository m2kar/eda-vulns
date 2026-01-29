# Minimization Report

## Summary
| Metric | Original | Minimized | Reduction |
|--------|----------|-----------|-----------|
| Lines | 18 | 2 | 88.9% |
| Characters | 411 | 32 | 92.2% |
| Constructs | 7 | 1 | 85.7% |

## Minimization Strategy

### Key Insight from Root Cause Analysis
The crash is triggered by `inout wire` port type producing `llhd.ref<i1>` in the IR, which the Arc dialect's `StateType::get()` cannot handle. All other constructs (always_comb, tristate assignment, macros, etc.) are irrelevant to the crash.

### Removed Constructs
- ✅ `define WIDTH 8` - macro definition (not needed)
- ✅ `input logic a` - input port (not needed)
- ✅ `output logic b` - output port (not needed)
- ✅ `input logic [`WIDTH-1:0] data_in` - multi-bit input (not needed)
- ✅ `always_comb begin ... end` - combinational block (not needed)
- ✅ `assign c = a ? data_in[0] : 1'bz` - tristate assignment (not needed)

### Preserved Constructs
- ✅ `module M(...)` - module declaration (required)
- ✅ `inout wire c` - bidirectional port (CRITICAL: triggers the bug)

## Verification

### Original Crash Signature
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
...LowerState.cpp:219:66
```

### Minimized Crash Signature
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
...LowerState.cpp:219:66
```

**Match: ✅ 100% identical error message and crash location**

## Minimization Steps

### Step 1: Aggressive Reduction
Hypothesis: Only the `inout wire` port is needed.
```systemverilog
module M(inout wire c);
endmodule
```
Result: **CRASH REPRODUCED** ✅

No further minimization possible - this is the minimal reproducer.

## Final Minimized Test Case

```systemverilog
module M(inout wire c);
endmodule
```

## Reproduction Command
```bash
circt-verilog --ir-hw bug.sv | arcilator
```

## Notes
- The `inout wire` keyword combination is the sole trigger for this bug
- The crash occurs regardless of any usage of the inout port
- Simply declaring an inout port is sufficient to cause the assertion failure
