# [Arc] Assertion failure when lowering inout ports in sequential logic

## Summary

**Status**: Bug appears to be **FIXED** in current CIRCT version

This issue documents an assertion failure in the arcilator's LowerState pass when processing SystemVerilog modules with inout (bidirectional) ports. The crash occurred because `computeLLVMBitWidth()` in `ArcTypes.cpp` did not handle `llhd::RefType`, which is used for inout port representations.

**Test Case**: `260129-00001875`

## Crash Details

### Original Error (CIRCT 1.139.0)

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: .../lib/Dialect/Arc/Transforms/LowerState.cpp:219: ...:
Assertion `succeeded(ConcreteT::verifyInvariants(...))` failed.
Stack dump:
  #0 ... llvm::sys::PrintStackTrace(llvm::raw_ostream&, int)
  #11 circt::arc::StateType::get(mlir::Type)
  #12 (anonymous namespace)::ModuleLowering::run()
  #13 (anonymous namespace)::LowerStatePass::runOnOperation()
```

### Crash Location

| Component | File | Line | Function |
|-----------|-------|-------|----------|
| **Assertion** | `lib/Dialect/Arc/ArcTypes.cpp` | 79 | `StateType::verify` |
| **Caller** | `lib/Dialect/Arc/Transforms/LowerState.cpp` | 219 | `ModuleLowering::run` |

### Error Message

```
state type must have a known bit width; got '!llhd.ref<i1>'
```

## Test Case

### Minimal Test Case (bug.sv)

```systemverilog
// Minimal test case for arcilator inout port crash
// Original bug: StateType cannot handle !llhd.ref<i1> from inout port
module test(inout wire c);
endmodule
```

### Original Test Case (source.sv)

```systemverilog
module test_module(inout wire c, input logic a);
  logic [3:0] temp_reg;
  
  initial begin
    temp_reg = 4'b1010;
  end
  
  assign c = (a) ? temp_reg[0] : 1'bz;
endmodule
```

### Reproduction Command

```bash
export PATH=/opt/llvm-22/bin:$PATH
circt-verilog --ir-hw bug.sv | arcilator | opt -O0 | llc -O0 --filetype=obj
```

## Root Cause Analysis

### Problem Description

The arcilator's `LowerState` pass crashes when processing modules containing inout (bidirectional) ports:

1. **Frontend** (`circt-verilog --ir-hw`) correctly converts inout port to `!llhd.ref<i1>` type
2. **LowerState pass** attempts to create `StateType` for module input arguments
3. **StateType::verify()** calls `computeLLVMBitWidth()` to get bit width
4. **Missing type support**: `computeLLVMBitWidth()` does not handle `llhd::RefType`
5. **Assertion failure**: Returns failure because no bit width can be computed for reference type

### Affected Components

- **Tool**: `arcilator`
- **Pass**: `arc-lower-state`
- **Dialect**: Arc/LLHD
- **Function**: `StateType::get()`

### Type Analysis

| Type | Description | Bit Width Support |
|-------|-------------|------------------|
| `!llhd.ref<i1>` | LLHD reference type for inout ports | ❌ Missing |
| `!llhd.i1` | LLHD integer type | ✅ Supported |
| `!arc.state<T>` | Arc state type wrapper | ✅ Supported |

## Verification Results

### Current Status (2026-02-01)

| Test | Result |
|-------|--------|
| **Crash Reproduced** | ❌ **NO** - Bug appears to be fixed |
| **Syntax Check** | ✅ Passed (Slang) |
| **CIRCT Compilation** | ✅ Success - generates `!llhd.ref<i1>` type |
| **Arcilator Pipeline** | ✅ Success - no crash or assertion failure |

### Pipeline Status

```
✓ circt-verilog: Generated hw.module with inout wire typed as !llhd.ref<i1>
✓ arcilator:      Converted to valid LLVM IR without assertion failure
✓ opt -O0:        LLVM IR optimization succeeded
✓ llc -O0:        Generated 784-byte object file
```

## Related Issues

### Duplicate Found

- **Issue #9574**: "[Arc] Assertion failure when lowering inout ports in sequential logic"
- **Status**: OPEN
- **Similarity**: 7.0/10.0
- **Matching Keywords**: `LowerState`, `Arc`, `StateType`, `inout`, `assertion failure`

### Comparison

| Aspect | Original Issue | This Test Case |
|--------|----------------|----------------|
| Error Message | Identical | Identical |
| Crash Location | LowerState.cpp:219 | LowerState.cpp:219 |
| Dialect | Arc | Arc |
| Trigger | inout port in sequential logic | inout port |

## Recommendations

### For Maintainers

1. **Close as duplicate**: This test case is likely a duplicate of Issue #9574
2. **Verify fix coverage**: Ensure existing fix addresses this test case variant
3. **Update documentation**: Add inout port support notes to Arc dialect documentation

### For Users

1. **Update CIRCT**: Use latest version which includes the fix
2. **Monitor Issue #9574**: Track resolution status

## Metadata

| Field | Value |
|-------|-------|
| **Test Case ID** | 260129-00001875 |
| **CIRCT Version** | 1.139.0 (original), Current (verified) |
| **LLVM Version** | 22.0.0git |
| **Classification** | not_a_bug (fixed) |
| **Minimization** | 55.6% reduction (9 lines → 4 lines) |
| **Analyzed Date** | 2026-02-01 |

## Analysis Files

- `root_cause.md` - Detailed root cause analysis
- `analysis.json` - Structured analysis data
- `validation.md` - Validation report
- `duplicates.md` - Duplicate check report
