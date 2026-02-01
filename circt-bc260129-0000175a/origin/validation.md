# Validation Report

## Test Information

| Field | Value |
|-------|-------|
| Testcase | bug.sv |
| Original Size | 22 lines |
| Minimized Size | 5 lines |
| Reduction | 77% |

## Syntax Validation

### slang Verification

```
$ slang bug.sv
Top level design units:
    top

Build succeeded: 0 errors, 0 warnings
```

**Result**: ✅ Valid SystemVerilog syntax (verified by slang 10.0.6)

## Feature Analysis

| Feature | Standard | Section | Status |
|---------|----------|---------|--------|
| Class definition | IEEE 1800-2017 | 8. Classes | Standard |
| Class handle | IEEE 1800-2017 | 8.4 Object handles | Standard |
| always_ff block | IEEE 1800-2017 | 9.2.2.4 Sequential logic | Standard |
| new() operator | IEEE 1800-2017 | 8.7 Constructors | Standard |

All features used in the testcase are standard SystemVerilog features defined in IEEE 1800-2017.

## circt-verilog Behavior

### Command
```bash
/opt/firtool/bin/circt-verilog --ir-hw bug.sv
```

### Output
```
bug.sv:2:9: remark: Class builtin functions (needed for randomization, constraints, and covergroups) are not yet supported and will be dropped during lowering.
  class C; endclass
        ^
bug.sv:3:5: error: 'hw.bitcast' op result #0 must be Type wherein the bitwidth in hardware is known, but got '!llvm.ptr'
  C obj;
    ^
bug.sv:3:5: note: see current operation: %10 = "hw.bitcast"(%9) : (i1073741823) -> !llvm.ptr
```

### Error Analysis

1. **Remark**: Class builtin functions are not yet supported - this is expected and does not affect this bug
2. **Error**: `hw.bitcast` operation fails because bitwidth is unknown for `!llvm.ptr` type
3. **Critical Value**: `i1073741823` indicates an invalid bitwidth (derived from -1 interpreted as large unsigned)

## Classification

### Decision: **REPORT** (Bug)

**Reasoning**:

1. **Syntax is valid**: slang successfully parses the code with no errors or warnings
2. **Standard features only**: All constructs used are defined in IEEE 1800-2017
3. **Tool failure**: circt-verilog fails with an internal error, not a proper "feature not supported" message
4. **Root cause**: The error stems from improper handling of `ClassHandleType` in the type system
   - `hw::getBitWidth()` returns -1 for unsupported types
   - This -1 propagates and causes invalid operations

### Why not "feature_request"?

The remark about "Class builtin functions not yet supported" refers to randomization, constraints, and covergroups - not basic class instantiation. The fundamental class handle and new() operator should work for the lowering to succeed, but instead produces an internal error.

### Why not "not_a_bug"?

- The code is valid SystemVerilog
- Other tools (slang) handle it correctly
- circt-verilog crashes/errors rather than gracefully declining to support the feature

## Summary

| Field | Value |
|-------|-------|
| **Classification** | report |
| **Bug Type** | Internal error / Type system bug |
| **Severity** | High |
| **Affected Component** | HW bitwidth handling for ClassHandleType |
| **Reproducibility** | Deterministic |
