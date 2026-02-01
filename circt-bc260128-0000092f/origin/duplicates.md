# Duplicate Issue Check Report

## Crash Summary
- **Dialect**: SV (SystemVerilog) → HW/Arc lowering
- **Crash Type**: Assertion failure in `llvm::cast<mlir::IntegerType>`
- **Component**: `InferStateProperties.cpp:211` (applyEnableTransformation)
- **Root Cause**: Type mismatch when creating hw.constant - enable signal type is non-IntegerType

## Search Strategy
Conducted 10 search queries focusing on:
- Arc dialect specific crashes (InferStateProperties, applyEnableTransformation)
- Type casting assertions (IntegerType, cast assertion)
- HW/SV lowering issues (hw.constant, hw::ConstantOp::create)
- Arcilator-specific lowering failures
- Enable transformation and state properties

## Top Findings

### 🔴 MOST SIMILAR - Issue #9467
**Title**: [circt-verilog][arcilator] `arcilator` fails to lower `llhd.constant_time` generated from simple SV delay (`#1`)
- **State**: OPEN
- **Similarity Score**: 13/30
- **Matching Components**: arcilator + SV/Verilog input
- **Reasoning**: Directly involves arcilator processing Verilog input with lowering failures. While the specific error (constant_time lowering) differs from our type casting issue, both represent failures in the arcilator's lowering pipeline when processing SV constructs.
- **URL**: https://github.com/llvm/circt/issues/9467

### 🟡 SECONDARY - Issue #8286
**Title**: [circt-verilog][llhd][arcilator] Verilog-to-LLVM lowering issues
- **State**: OPEN  
- **Similarity Score**: 10/30
- **Matching Components**: arcilator + Verilog input
- **Reasoning**: Generic Verilog-to-LLVM lowering issue in arcilator. Broader scope than our specific cast assertion, but same pipeline affected.
- **URL**: https://github.com/llvm/circt/issues/8286

### 🟡 SECONDARY - Issue #8930
**Title**: [MooreToCore] Crash with sqrt/floor
- **State**: OPEN
- **Similarity Score**: 10/30
- **Matching Components**: IntegerType assertion
- **Reasoning**: Another assertion crash involving IntegerType, but in MooreToCore (not Arc/applyEnableTransformation). Different error path.
- **URL**: https://github.com/llvm/circt/issues/8930

## Conclusion

**Recommendation**: **REVIEW_EXISTING** (score: 13 ≥ 10 threshold)

While Issue #9467 has the highest similarity score, it appears to be about `llhd.constant_time` lowering failures rather than enable signal type casting. However, both involve:
- arcilator processing SV inputs
- Type/lowering failures in the conversion pipeline
- OPEN status

**Action**: The issue should be reviewed against #9467 to determine if they represent the same underlying bug or distinct problems in arcilator's SV lowering path.

## Other Issues Considered
- #8024 (Comb folder crash) - Score 4 - Generic assertion, unrelated
- #8844 (moore.case_eq type error) - Score 2 - Array vs simple vector, different dialect

## Keywords Searched
InferStateProperties, Arc applyEnableTransformation, hw::ConstantOp::create cast assertion, IntegerType cast assertion, Arc dialect, arcilator, hw.constant type assertion, enable transformation, state properties, SV to HW lowering
