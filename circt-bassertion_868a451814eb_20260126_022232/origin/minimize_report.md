# Minimization Report

## Summary

| Metric | Value |
|--------|-------|
| **Original file** | source.sv (7 lines) |
| **Minimized file** | bug.sv (1 line) |
| **Reduction** | 85.7% |
| **Crash preserved** | Yes |

## Minimization Process

### Original Test Case
```systemverilog
module example(output string out);
  string str;
  always_comb begin
    str = "Hello";
    out = str;
  end
endmodule
```

### Minimized Test Case
```systemverilog
module e(output string o);endmodule
```

### Key Findings

The crash is triggered by the **minimum construct**: a module with a `string` type output port. 

All other elements are non-essential:
- ❌ `always_comb` block - not required
- ❌ Local `string` variable - not required  
- ❌ String assignment - not required
- ❌ Module body - not required
- ✅ `output string` port - **essential trigger**

### Verification

| Step | Description | Result |
|------|-------------|--------|
| 1 | Remove `always_comb` block | Crash preserved |
| 2 | Remove local variable `str` | Crash preserved |
| 3 | Shorten module name | Crash preserved |
| 4 | Shorten port name | Crash preserved |
| 5 | Single-line format | Crash preserved |

### Crash Signature Comparison

**Original:**
```
SVModuleOpConversion::matchAndRewrite ... MooreToCore.cpp
```

**Minimized:**
```
SVModuleOpConversion::matchAndRewrite ... MooreToCore.cpp
```

**Match**: ✅ Exact match

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv
```

## Root Cause Confirmation

The minimization confirms the root cause hypothesis from `analysis.json`:
- **Problem**: `string` type in output port context
- **Mechanism**: Type conversion returns null for `string` type, causing `dyn_cast` assertion failure
- **Location**: `MooreToCore.cpp` in `SVModuleOpConversion::matchAndRewrite`

## Output Files

| File | Description |
|------|-------------|
| `bug.sv` | Minimized test case (1 line) |
| `error.log` | Crash output with stack trace |
| `command.txt` | Single-line reproduction command |
