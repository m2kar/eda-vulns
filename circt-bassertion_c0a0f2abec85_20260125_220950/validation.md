# Validation Report

## Test Case
**File:** bug.sv
```systemverilog
module M(inout logic c);
  assign c = 0;
endmodule
```

## Syntax Validation Results

| Tool | Version | Command | Result | Notes |
|------|---------|---------|--------|-------|
| Verilator | latest | `verilator --lint-only bug.sv` | ✅ Pass | No errors or warnings |
| Slang | latest | `slang --lint-only bug.sv` | ✅ Pass | 0 errors, 0 warnings |
| Icarus Verilog | latest | `iverilog -g2012 -o /dev/null bug.sv` | ✅ Pass | No errors |

**Conclusion:** The testcase is syntactically valid SystemVerilog per IEEE 1800-2017.

## Feature Support Analysis

### `inout` Port Declaration

| Standard/Tool | Support Status |
|---------------|----------------|
| IEEE 1800-2017 | ✅ Supported (Section 23.2.2.3) |
| Verilator | ✅ Supported |
| Slang | ✅ Supported |
| Icarus Verilog | ✅ Supported |
| CIRCT arcilator | ❌ Not supported (crashes) |

## CIRCT Crash Verification

**Tool Chain:**
```bash
/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw bug.sv | \
/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/arcilator
```

**Result:** 💥 Assertion failure

**Error Message:**
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
Assertion `succeeded(ConcreteT::verifyInvariants(...))' failed.
```

## Classification

| Field | Value |
|-------|-------|
| **Result** | `report` |
| **Bug Type** | Missing validation |
| **Severity** | Medium |
| **Impact** | arcilator crashes instead of providing user-friendly error |

## Rationale

1. **Valid Input:** The testcase passes all three industry-standard SystemVerilog tools (Verilator, Slang, Icarus)
2. **Standard Compliance:** `inout` ports are a valid IEEE 1800-2017 construct
3. **Improper Handling:** arcilator crashes with assertion failure instead of emitting a diagnostic error
4. **User Impact:** Users get cryptic crash message instead of clear "inout ports not supported" error

## Recommendation

This should be reported as a bug. The fix should:
1. Add early validation in `LowerState` pass to detect `llhd::RefType` (inout) arguments
2. Emit user-friendly error: `"arcilator does not support inout ports"`
3. Fail gracefully instead of assertion crash
