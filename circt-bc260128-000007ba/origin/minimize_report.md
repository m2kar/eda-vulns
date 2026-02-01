# Minimization Report

## Summary

- **Original file**: source.sv (18 lines)
- **Minimized file**: bug.sv (2 lines)
- **Reduction**: 88.9%
- **Crash preserved**: Yes

## Preservation Analysis

### Key Constructs Preserved

Based on `analysis.json`, the following constructs were kept:
- `inout logic x` - bidirectional port (critical for triggering crash)

### Removed Elements

| Element | Lines Removed | Reason |
|---------|---------------|--------|
| `input logic clk` | 1 | Not related to crash |
| `input logic [63:0] wide_input` | 1 | Not related to crash |
| `output logic [31:0] out_val` | 1 | Not related to crash |
| `integer idx` | 1 | Not related to crash |
| `always_ff` block | 3 | Not related to crash |
| `always_comb` block | 3 | Not related to crash |
| Module name shortened | - | `MixPorts` → `M` |
| Port name shortened | - | `io_sig` → `x` |

## Verification

### Original Assertion
```
state type must have a known bit width; got '!llhd.ref<i1>'
```

### Final Assertion
```
state type must have a known bit width; got '!llhd.ref<i1>'
```

**Match**: ✅ Exact match

## Reproduction Command

```bash
/home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw bug.sv | /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/arcilator
```

## Minimized Test Case

```systemverilog
module M(inout logic x);
endmodule
```

## Notes

- The crash is solely caused by the `inout` port declaration
- The `inout` port is lowered to `!llhd.ref<i1>` type in MLIR
- The Arc dialect's `StateType` cannot handle LLHD reference types
- All other module content (clk, data ports, always blocks) was irrelevant
