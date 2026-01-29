# Validation Report

## Syntax Validation
- **Status**: ✅ Valid SystemVerilog
- **Standard**: IEEE 1800-2017 Section 6.16 (String data type)

## Classification

| Field | Value |
|-------|-------|
| Is Bug | ❌ No |
| Category | Unsupported Feature |
| Related Issue | [#8332](https://github.com/llvm/circt/issues/8332) |

## Explanation

The test case uses `string` type as a module port, which is valid SystemVerilog syntax per IEEE 1800-2017. However, `string` is a dynamic simulation type and not synthesizable to hardware.

The MooreToCore lowering pass in CIRCT does not yet support converting `StringType` to the HW/Core dialects, which is a known limitation tracked in issue #8332.

## Test Minimization

| Metric | Value |
|--------|-------|
| Original | 26 lines |
| Minimized | 2 lines |
| Reduction | 92.3% |

### Minimized Test Case
```systemverilog
module m(input string s);
endmodule
```

## Recommendation

**Do not submit new issue** - This crash is a duplicate of the existing feature request #8332.
