# Validation Report

## Summary

| Check | Result |
|-------|--------|
| Syntax Check | ✅ valid |
| Feature Support | ✅ supported |
| Known Limitations | ⚠️ matched (#6343) |
| **Classification** | **report** (duplicate) |

## Minimization Results

| Metric | Value |
|--------|-------|
| Original | 22 lines |
| Minimized | 11 lines |
| Reduction | **50.0%** |

### Key Constructs Preserved
- `affine.for` loop → converted to `scf.while`
- `func.call @ext()` → external function call in loop body

## Syntax Validation

**Tool**: mlir-opt  
**Status**: ✅ Valid

The test case is valid MLIR and successfully processes through:
1. `mlir-opt` - Parses correctly
2. `mlir-opt --lower-affine` - Converts to SCF dialect
3. `mlir-opt --lower-affine --scf-for-to-while` - Converts to while loop

## Cross-Tool Validation

| Pass | Result |
|------|--------|
| mlir-opt (parse) | ✅ pass |
| --lower-affine | ✅ pass |
| --scf-for-to-while | ✅ pass |
| circt-opt --lower-scf-to-calyx | ❌ **crash** |

## Root Cause

The crash occurs in `circt::scftocalyx::BuildControl::buildCFGControl` when encountering a `func.call` operation inside an SCF while loop body. The `SuccessorRange` constructor receives a null block pointer, causing a segmentation fault.

**Stack trace key frames:**
```
mlir::SuccessorRange::SuccessorRange(mlir::Block*)
circt::scftocalyx::BuildControl::buildCFGControl(...)
circt::scftocalyx::BuildControl::partiallyLowerFuncToComp(...)
```

## Known Limitation Match

**GitHub Issue**: [#6343](https://github.com/llvm/circt/issues/6343)  
**Title**: MLIR lowering issue  
**Status**: Open  
**Similarity**: 95%

The maintainer confirmed: *"there is an actual bug somewhere in scf-to-calyx"*

## Classification

**Result**: `report`  
**Is Duplicate**: Yes (of #6343)

### Reasoning
This is a valid MLIR test case that causes a segmentation fault in CIRCT's SCFToCalyx pass. The crash is reproducible and matches the pattern of known issue #6343. While this is a duplicate, the minimized test case (11 lines) may be useful as an additional reproducer.

## Recommendation

✅ This is a confirmed bug in CIRCT.  
⚠️ **Already reported as #6343** - consider adding this minimized test case as a comment to the existing issue rather than filing a new one.

## Reproduction

```bash
mlir-opt --lower-affine --scf-for-to-while bug.mlir | \
  /opt/firtool-1.139.0/bin/circt-opt --pass-pipeline='builtin.module(lower-scf-to-calyx{top-level-function=f})'
```
