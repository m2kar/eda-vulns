# Minimize Report

## Summary
- **Original testcase**: 25 lines (source.sv)
- **Minimized testcase**: 3 lines (bug.sv)
- **Reduction**: 88%

## Minimization Process

### Step 1: Analyze key constructs from analysis.json
- Key feature: `inout port` declaration
- Crash trigger: `!llhd.ref` type cannot be processed by arcilator's LowerState pass

### Step 2: Iterative reduction

| Iteration | Modification | Result | Lines |
|-----------|--------------|--------|-------|
| 1 | Remove always_ff, always_comb, extra ports | Crash reproduced | 3 |
| 2 | Remove vector width [7:0] | Crash reproduced (i1 type) | 3 |
| 3 | Simplify assign to `c = 0` | Crash reproduced | 3 |
| 4 | Remove assign statement | No crash (empty module) | 2 |

### Step 3: Final minimized testcase

```systemverilog
module M(inout logic c);
  assign c = 0;
endmodule
```

## Crash Verification

**Command:**
```bash
/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw bug.sv | /edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/arcilator
```

**Error signature:**
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
Assertion `succeeded(ConcreteT::verifyInvariants(...))' failed.
```

**Stack trace match:** ✓
- `circt::arc::StateType::get(mlir::Type)` at ArcTypes.cpp.inc:108
- `ModuleLowering::run()` at LowerState.cpp:219

## Key Findings

1. The crash is triggered by **any inout port** with an assign statement
2. The port width (i1 vs i8) does not affect the crash occurrence
3. The assign value (0, 8'bz, or expression) does not affect the crash
4. An empty inout port without assign does NOT trigger the crash (module compiles)

## Root Cause Confirmation

The minimized testcase confirms the analysis.json hypothesis:
- arcilator's `LowerState` pass encounters `!llhd.ref<i1>` type from inout port
- `StateType::get()` cannot compute bit width for `llhd::RefType`
- Assertion failure occurs instead of user-friendly error
