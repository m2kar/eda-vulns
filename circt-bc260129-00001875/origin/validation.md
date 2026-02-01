# Validation Report

## Test Case Information
- **Original File**: source.sv (9 lines)
- **Minimized File**: bug.sv (4 lines)
- **Reduction**: 55.6%

## Minimal Test Case (bug.sv)
```systemverilog
// Minimal test case for arcilator inout port crash
// Original bug: StateType cannot handle !llhd.ref<i1> from inout port
module test(inout wire c);
endmodule
```

## Validation Results

### Syntax Validation
- **Tool**: slang --lint-only
- **Result**: ✅ PASSED (0 errors, 0 warnings)

### CIRCT Compilation
- **Tool**: circt-verilog --ir-hw
- **Result**: ✅ SUCCESS
- **Generated IR**:
```mlir
module {
  hw.module @test(in %c : !llhd.ref<i1>) {
    hw.output
  }
}
```

### Arcilator Pipeline
- **Tool**: circt-verilog --ir-hw | arcilator
- **Result**: ✅ SUCCESS (no crash)
- **Output**: Valid LLVM IR generated

## Bug Status

### Original Bug (CIRCT 1.139.0)
- **Error**: `state type must have a known bit width; got '!llhd.ref<i1>'`
- **Cause**: `computeLLVMBitWidth()` in ArcTypes.cpp did not handle `llhd::RefType`
- **Crash Location**: `StateType::get()` → `StateType::verify()` assertion failure

### Current Status
- **Reproduces**: ❌ NO
- **Classification**: `not_a_bug` (fixed in current version)

## Conclusion

The original bug where arcilator crashed with an assertion failure when processing modules with inout (bidirectional) ports has been **fixed** in the current CIRCT version. The `!llhd.ref<i1>` type representing inout ports is now handled correctly by the arcilator pipeline.

The minimal test case preserves the essential construct (inout port) that originally triggered the bug for historical documentation purposes.

## Files Generated
- `bug.sv` - Minimal test case
- `error.log` - Original error details
- `command.txt` - Reproduction commands
- `validation.json` - Structured validation data
- `validation.md` - This report
