# Minimization Report

## Summary

| Metric | Value |
|--------|-------|
| **Original file** | source.sv (18 lines) |
| **Minimized file** | bug.sv (2 lines) |
| **Reduction** | 88.9% |
| **Crash preserved** | ✅ Yes |

## Key Constructs Preserved

Based on `analysis.json`, the following constructs were identified as essential:

| Construct | Status |
|-----------|--------|
| `output string port` | ✅ Preserved |
| `string literal assignment` | ❌ Removed (not required for crash) |
| `string method call (len)` | ❌ Removed (not required for crash) |
| `unpacked array` | ❌ Removed (not required for crash) |
| `always_comb` | ❌ Removed (not required for crash) |

## Minimization Process

### Step 1: Initial Analysis
- Identified core problem: `string type as module output port`
- Key construct from analysis.json: `output string a`

### Step 2: Iterative Removal
1. Removed `input logic [31:0] arg0` - crash preserved ✅
2. Removed `logic [31:0] arr [0:3]` - crash preserved ✅
3. Removed `assign arr[0] = arg0[31:0]` - crash preserved ✅
4. Removed `always_comb` block - crash preserved ✅
5. Removed `int length = a.len()` - crash preserved ✅
6. Removed comments - crash preserved ✅

### Step 3: Final Verification
- Minimal reproduction: `module test_module(output string a); endmodule`
- Crash signature: `SVModuleOpConversion::matchAndRewrite` in MooreToCore.cpp

## Crash Signature Comparison

### Original Error
```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed
```

### Minimized Error
```
SVModuleOpConversion::matchAndRewrite ... MooreToCore.cpp
```

**Match**: ✅ Same crash location (MooreToCore.cpp, SVModuleOpConversion)

## Removed Elements

| Element | Lines Removed |
|---------|---------------|
| Input port declaration | 1 |
| Local array declaration | 1 |
| Assign statement | 1 |
| always_comb block | 6 |
| String method call | 2 |
| Comments | 2 |
| Empty lines | 3 |

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv
```

## Output Files

- `bug.sv` - Minimized test case (2 lines)
- `error.log` - Error output from minimized test
- `command.txt` - Single-line reproduction command
