# PR Title

[MooreToCore] Add UnionType conversion support (Fixes #9570)

---

# PR Description

## Summary

This PR adds complete support for SystemVerilog union type conversion in the MooreToCore pass, fixing a crash when processing union types with `circt-verilog`.

**Fixes #9570**

## Problem

When processing SystemVerilog code containing union types, `circt-verilog --ir-hw` would crash with an assertion failure because the MooreToCore conversion pass lacked support for union type conversion.

**Example crash (from #9570):**
```bash
$ circt-verilog --ir-hw bug.sv
circt-verilog: assertion failed
```

## Solution

This PR implements:

1. **Type Converters**: Added converters for `UnionType` and `UnpackedUnionType` to `hw::UnionType`
2. **Operation Converters**: Added conversion patterns for:
   - `UnionCreateOp` → `hw::UnionCreateOp`
   - `UnionExtractOp` → `hw::UnionExtractOp`
   - `UnionExtractRefOp` → `llhd::SigStructExtractOp`
3. **Bug Fix**: Fixed `UnionCreateOp::verify()` which was incorrectly comparing `member.type` with `resultType` instead of `inputType`
4. **LLHD Support**: Extended `SigStructExtractOp` to support both struct and union types
5. **Test Coverage**: Added comprehensive union conversion test in `basic.mlir`

## Changes

### Files Modified

- **include/circt/Dialect/LLHD/LLHDOps.td**: Updated type constraint to accept both structs and unions
- **lib/Conversion/MooreToCore/MooreToCore.cpp**: Added union type converters and operation patterns
- **lib/Dialect/LLHD/IR/LLHDOps.cpp**: Extended `SigStructExtractOp` for union support
- **lib/Dialect/Moore/MooreOps.cpp**: Fixed verifier bug in `UnionCreateOp`
- **test/Conversion/MooreToCore/basic.mlir**: Added union conversion test

### Code Statistics

- **Lines Added**: 158
- **Lines Removed**: 15
- **Files Modified**: 5

## Testing

### New Tests
- Added `@Union` test module covering all union operations
- Tests `union_create`, `union_extract`, and `union_extract_ref`
- Includes both value and reference operations

### Verification
```bash
# All existing tests pass
$ ninja -C build check-circt
✅ All tests pass

# Original bug is fixed
$ ./build/bin/circt-verilog --ir-hw bug.sv
✅ Successfully converts without crash
```

## Design Decisions

1. **Mirrored Struct Patterns**: Union converters follow the same structure as existing struct converters for consistency
2. **Reused LLHD Ops**: `SigStructExtractOp` now handles both structs and unions rather than creating a separate union-specific op
3. **Zero Offset**: Packed unions use offset=0 for all fields (standard union semantics)

## Backward Compatibility

✅ All changes are backward compatible:
- Existing struct operations unchanged
- LLHD ops gracefully handle both types
- No API changes

## Related Issues

Fixes #9570 - Crash when processing SystemVerilog union types in MooreToCore conversion.

---

## Checklist

- [x] Added comprehensive test coverage
- [x] All existing tests pass
- [x] Code follows CIRCT conventions
- [x] Code formatted with clang-format
- [x] No debug output or unnecessary changes
- [x] Backward compatible

