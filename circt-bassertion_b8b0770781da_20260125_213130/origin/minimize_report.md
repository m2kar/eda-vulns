# Test Case Minimization Report

## Summary

| Metric | Value |
|--------|-------|
| Original Lines | 15 |
| Minimized Lines | 2 |
| Reduction | **86.7%** |
| Crash Preserved | ✅ Yes |

## Original Test Case

```systemverilog
module string_register(
  input logic clk,
  input string data_in,
  output string data_out
);

  string a = "Test";
  
  assign data_in = a;
  
  always @(posedge clk) begin
    data_out <= data_in;
  end

endmodule
```

## Minimized Test Case

```systemverilog
module test(input string s);
endmodule
```

## Minimization Strategy

Based on `analysis.json`, the essential construct is:
- **string type port declaration** (`input string` or `output string`)

### Removed Elements

| Element | Necessary | Reason |
|---------|-----------|--------|
| `input logic clk` | ❌ No | Not related to string type handling |
| `output string data_out` | ❌ No | Single string port is sufficient |
| `string a = "Test"` | ❌ No | Internal variables not needed |
| `assign data_in = a` | ❌ No | Assignment not needed |
| `always @(posedge clk)` | ❌ No | Procedural block not needed |
| Module name | ❌ No | Simplified to `test` |
| Port name | ❌ No | Simplified to `s` |

### Preserved Elements

| Element | Reason |
|---------|--------|
| `module test(...)` | Module declaration required |
| `input string s` | **Core trigger**: string type as port |
| `endmodule` | Module end required |

## Verification Steps

1. **Step 1**: Remove `input logic clk` → Crash ✅
2. **Step 2**: Remove `output string data_out` → Crash ✅
3. **Step 3**: Remove internal `string a` variable → Crash ✅
4. **Step 4**: Remove `assign` statement → Crash ✅
5. **Step 5**: Remove `always` block → Crash ✅
6. **Step 6**: Try `output string` only → Crash ✅ (both work)
7. **Final**: Single line `input string s` → Crash ✅

## Crash Signature Match

| Field | Original | Minimized |
|-------|----------|-----------|
| Crash Type | assertion | assertion |
| Pass | MooreToCore | MooreToCore |
| Function | SVModuleOpConversion | SVModuleOpConversion |
| Message | dyn_cast on a non-existent value | dyn_cast on a non-existent value |

**Match**: ✅ Same crash path confirmed

## Conclusion

The crash is triggered by the **minimal construct**: a module with a `string` type port. No other code elements are necessary to reproduce the bug.

### Root Cause Connection

From `analysis.json`:
1. Moore dialect converts `StringType` to `sim::DynamicStringType`
2. HW dialect does not support `DynamicStringType` as port type
3. `dyn_cast<InOutType>` fails on incompatible type

The minimized test case isolates exactly this issue - the mere presence of a `string` port triggers the type conversion failure.
