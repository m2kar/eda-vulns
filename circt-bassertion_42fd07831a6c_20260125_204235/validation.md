# Validation Report

## Summary
| Aspect | Result |
|--------|--------|
| **Syntax Valid** | ✅ Yes |
| **Classification** | 🐛 Compiler Bug |
| **Confidence** | High |
| **Recommendation** | Report to CIRCT |

## Test Case
```systemverilog
module test(output string str);
endmodule
```

## Cross-Tool Validation

### Verilator (5.022)
- **Result**: ✅ Pass
- **Command**: `verilator --lint-only bug.sv`
- **Errors**: None
- **Warnings**: None

### Slang (10.0.6)
- **Result**: ✅ Pass  
- **Command**: `slang --lint-only bug.sv`
- **Errors**: None
- **Warnings**: None
- **Output**: `Build succeeded: 0 errors, 0 warnings`

### CIRCT (firtool-1.139.0)
- **Result**: ❌ Crash
- **Command**: `circt-verilog --ir-hw bug.sv`
- **Crash Type**: Assertion failure
- **Assertion**: `detail::isPresent(Val) && "dyn_cast on a non-existent value"`

## Analysis

### Language Feature
- **Construct**: `string` type module port
- **IEEE Standard**: IEEE 1800-2017
- **Synthesizable**: No (simulation-only)
- **Valid SystemVerilog**: Yes

### Bug Classification Reasoning

1. **Valid Syntax**: Both Verilator and Slang accept the test case without errors
2. **Crash Instead of Error**: CIRCT should emit a proper error message if string ports are unsupported, not crash with an assertion failure
3. **Internal Consistency**: The assertion failure indicates incomplete handling of a valid language construct

### Expected Behavior
If CIRCT does not support string type ports:
- Should emit: `error: string type ports are not supported`
- Should NOT crash with internal assertion failure

## Conclusion
This is a **genuine compiler bug** that should be reported. The test case uses valid SystemVerilog syntax, and CIRCT's crash represents a failure to handle unsupported features gracefully.

## Reproduction Command
```bash
circt-verilog --ir-hw bug.sv
```
