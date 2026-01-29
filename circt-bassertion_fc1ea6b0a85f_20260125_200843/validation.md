# Validation Report

## Summary

| Check | Result |
|-------|--------|
| Syntax Check (slang) | ✅ pass |
| Syntax Check (verilator) | ✅ pass |
| circt-verilog | ✅ accepted |
| arcilator | ✅ accepted (bug fixed) |
| **Classification** | **unsupported_feature** |

## Test Case

```systemverilog
module M(inout logic c);
endmodule
```

## Syntax Validation

### slang
```
Build succeeded: 0 errors, 0 warnings
```

### verilator
```
No errors
```

## Tool Compatibility

### circt-verilog --ir-hw

**Status**: Accepted

**Output**:
```mlir
module {
  hw.module @M(in %c : !llhd.ref<i1>) {
    hw.output
  }
}
```

Note: The `inout logic c` port is converted to `!llhd.ref<i1>` type.

### arcilator

**Status**: Accepted (Bug Fixed)

**Output**:
```llvm
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare void @exit(i32)

define void @M_eval(ptr %0) {
  ret void
}
```

The current CIRCT version handles the inout port gracefully by ignoring it in the simulation model.

## Classification

**Result**: `unsupported_feature`

**Confidence**: High

### Reasoning

1. **Arc Dialect Design Limitation**: Arc dialect does not support bidirectional (inout) ports by design
   - Confirmed in `ArcOps.cpp:338-339`: ModelOp verification checks for inout ports
   - Arc is designed for simulation, which requires unidirectional data flow

2. **Original Crash Analysis**:
   - Tool: arcilator
   - Pass: LowerState
   - Location: `LowerState.cpp:219`
   - Assertion: `state type must have a known bit width; got '!llhd.ref<i1>'`

3. **Root Cause**: The HW dialect represents inout ports as `!llhd.ref` type, but Arc's `StateType::get()` could not handle this type, leading to assertion failure.

4. **Bug Status**: **FIXED** in current CIRCT version
   - Current arcilator accepts the input and generates valid LLVM IR
   - The inout port is handled gracefully (ignored in simulation model)

## Recommendation

**No bug report needed**

### Reasons:
1. The crash has been fixed in the current CIRCT version
2. The underlying limitation (Arc not supporting inout) is a design constraint, not a bug
3. The fix likely added proper early validation or graceful handling

### Historical Value:
This test case documents that:
- Arc dialect has design limitations for inout ports
- `!llhd.ref` types were not properly handled in earlier versions
- Early validation was missing, causing assertion failures instead of user-friendly errors

## Files Generated

| File | Description |
|------|-------------|
| `bug.sv` | Minimized test case (2 lines) |
| `command.txt` | Reproduction command |
| `minimize_report.md` | Minimization process report |
| `validation.json` | Structured validation data |
| `validation.md` | This report |
