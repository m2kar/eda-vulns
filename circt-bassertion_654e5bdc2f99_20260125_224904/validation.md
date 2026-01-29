# Validation Report

## Summary

| Check | Result |
|-------|--------|
| Syntax Check | ✅ valid |
| Feature Support | ✅ supported |
| Known Limitations | ❌ none matched |
| **Classification** | **report** |

## Syntax Validation

**Tool**: slang  
**Status**: ✅ valid

```
Build succeeded: 0 errors, 0 warnings
```

## Feature Support Analysis

**Unsupported features detected**: None

The test case uses only standard SystemVerilog constructs:
- `module` declaration (IEEE 1800-2017 §23)
- `inout` port direction (IEEE 1800-2017 §23.2.2)

These are fundamental, widely-supported language features.

### CIRCT Known Limitations

No known limitation matched. However, this appears to be a gap in arcilator's support for bidirectional (inout) ports.

## Cross-Tool Validation

| Tool | Status | Notes |
|------|--------|-------|
| Slang | ✅ pass | Build succeeded: 0 errors, 0 warnings |
| Verilator | ✅ pass | No errors or warnings |
| Icarus Verilog | ✅ pass | Successfully compiled |

**Conclusion**: All three industry-standard SystemVerilog tools accept this code without any issues.

## Classification

**Result**: `report`  
**Confidence**: high

**Reasoning**:

1. **Valid Test Case**: The test case is valid IEEE 1800 SystemVerilog
2. **Cross-Tool Agreement**: All major EDA tools (Slang, Verilator, Icarus) accept this code
3. **Unique CIRCT Issue**: Only CIRCT's arcilator fails on this code
4. **Not a Feature Request**: `inout` ports are a fundamental language feature, not an obscure edge case

## Bug Details

| Attribute | Value |
|-----------|-------|
| Component | arcilator |
| Failing Pass | LowerState |
| Root Cause | `inout` ports converted to `!llhd.ref` type not supported by `arc::StateType::get()` |
| Error Type | Legalization failure |

### Error Message

```
error: failed to legalize operation 'arc.state_write'
  hw.module @M(in %d : !llhd.ref<i1>) {
... (!arc.state<!llhd.ref<i1>>, !llhd.ref<i1>) -> ()
```

## Recommendation

**Proceed to check for duplicates and generate the bug report.**

This is a valid bug in arcilator. The tool should either:
1. Support `inout` ports properly in the LowerState pass
2. Emit a clear, early error message indicating inout ports are not supported (instead of crashing during legalization)

## Minimized Test Case

```systemverilog
module M(inout d);
endmodule
```

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv | arcilator --observe-ports
```
