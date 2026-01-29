# Minimization Report

## Summary
- **Original file**: source.sv (14 lines)
- **Minimized file**: bug.sv (2 lines)
- **Reduction**: 85.7%
- **Crash preserved**: Yes

## Preservation Analysis

### Key Constructs Preserved
Based on `analysis.json`, the following constructs were kept:
- `output string str_out` - string type used as module port (core trigger)

### Removed Elements
- `input clk` - irrelevant to crash
- `string str` internal variable - not needed
- Conditional compilation block (`ifdef/else/endif`) - not needed
- `always` block - not needed
- `assign` statements - not needed

## Verification

### Original Assertion
```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed
```

### Final Assertion
Same crash in `SVModuleOpConversion::matchAndRewrite` at `MooreToCore.cpp`

**Match**: ✅ Exact match (same crash location and mechanism)

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv
```

## Analysis

The crash is triggered by the **string type** being used as a **module port**. The minimal trigger is:

```systemverilog
module top_module(output string str_out);
endmodule
```

This is the absolute minimal test case because:
1. `string` type ports are not valid for hardware synthesis
2. MooreToCore pass cannot convert `sim::DynamicStringType` to hw dialect types
3. The `dyn_cast<hw::InOutType>` fails on the unsupported type

## Notes
- The original test had conditional compilation and internal logic that were irrelevant
- The core issue is that CIRCT accepts `string` type ports but crashes during lowering
- Expected behavior: Either reject at parse time or emit proper error, not crash
