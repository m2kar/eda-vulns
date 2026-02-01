# Validation Report

## Summary

| Aspect | Result |
|--------|--------|
| **Classification** | `report` - Valid bug to report |
| **Syntax Valid** | ✓ Yes |
| **Cross-tool Verification** | ✓ Pass (slang, verilator) |
| **Reduction** | 62.5% (24 → 9 lines) |

## Test Case Validity

### Slang (v10.0.6)
```
Top level design units:
    m

Build succeeded: 0 errors, 0 warnings
```
**Result**: ✓ PASS

### Verilator (v5.022)
```
Exit code: 0
```
**Result**: ✓ PASS

### CIRCT Moore IR (firtool-1.139.0)
```mlir
module {
  moore.module @m(in %x : !moore.union<{a: l1}>) {
    moore.output
  }
}
```
**Result**: ✓ PASS - Moore IR correctly represents the packed union type

### CIRCT HW IR (MooreToCore)
```
PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
...
Assertion failure: detail::isPresent(Val) && "dyn_cast on a non-existent value"
```
**Result**: ✗ CRASH

## Classification Rationale

This is a **valid compiler bug** because:

1. **Valid input**: The SystemVerilog code is syntactically and semantically correct per IEEE 1800-2017
   - Packed unions are a standard SystemVerilog feature
   - Both commercial-grade tools (slang, verilator) accept the code

2. **Consistent behavior**: The crash is deterministic and reproducible with the same assertion failure

3. **Clear root cause**: Missing type conversion for `PackedUnionType` in MooreToCore pass
   - The type converter handles IntType, ArrayType, StructType, UnpackedStructType
   - But NOT PackedUnionType
   - This causes `convertType()` to return null, which propagates to an invalid dyn_cast

4. **Crash location**: `MooreToCore.cpp:SVModuleOpConversion::matchAndRewrite`
   - The crash occurs during dialect conversion, not parsing
   - This indicates a missing feature in the conversion pass

## Minimized Test Case

```systemverilog
// Minimal reproducer: packed union type as module port crashes MooreToCore pass
// Bug: Missing type conversion for PackedUnionType in MooreToCore

typedef union packed {
  logic a;
} u;

module m(input u x);
endmodule
```

## Recommendation

**Report this bug** to https://github.com/llvm/circt with:
1. The minimized test case (bug.sv)
2. The reproduction command: `circt-verilog --ir-hw bug.sv`
3. CIRCT version: firtool-1.139.0
4. Root cause: Missing PackedUnionType conversion in MooreToCore pass

## Fix Suggestion

Add a type conversion handler for `moore::PackedUnionType` in `lib/Conversion/MooreToCore/MooreToCore.cpp` in the `populateTypeConversion` function, similar to the existing handlers for `IntType`, `StructType`, etc.
