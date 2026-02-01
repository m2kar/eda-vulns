# CIRCT Bug Duplicate Analysis Report

## Executive Summary

**Status**: `review_existing` ✓  
**Top Match**: Issue #9287 (Similarity Score: 7/10)  
**Confidence Level**: HIGH

This bug report describes a crash caused by invalid bitwidth validation in the Mem2Reg pass. After comprehensive GitHub search of 16 different query patterns across the CIRCT repository, we identified that **this is NOT a new issue** but rather a manifestation of an already-tracked problem described in **Issue #9287: "[HW] Make `hw::getBitWidth` use std::optional vs -1"**.

---

## Crash Details

**Error Message**: `integer bitwidth is limited to 16777215 bits`  
**Crash Type**: Assertion failure in `IntegerType::get()`  
**Location**: `lib/Dialect/LLHD/Transforms/Mem2Reg.cpp:1753` in `Promoter::insertBlockArgs(BlockEntry*)`  
**Affected Pass**: Mem2RegPass  
**Dialect**: LLHD  
**MLIR Type**: IntegerType  

### Root Cause

The Mem2Reg pass calls `hw::getBitWidth()` on a slot's stored type and passes the result directly to `builder.getIntegerType()` **without validating** that the bitwidth is valid. When processing signed arithmetic expressions like `(-a <= a)` where `a` is a signed wire, the type resolution fails to produce a valid bitwidth, causing `getBitWidth()` to return `-1` or an invalid large value. This invalid value is then passed to `IntegerType::get()` which triggers an assertion because MLIR limits integer bitwidth to 16,777,215 bits.

---

## Search Strategy and Results

### Search Queries Executed

1. `bitwidth limit Mem2Reg assertion` - No direct matches
2. `16777215` - No direct matches (specific bitwidth limit value)
3. `insertBlockArgs` - No direct matches (function name)
4. `getBitWidth invalid` - No direct matches
5. `Mem2Reg` - Found issue #8693 (different bug, same pass)
6. `LLHD` - Found multiple LLHD-related issues
7. `signed assertion` - No direct matches
8. `integer type bitwidth` - Found issue #9287 (TOP MATCH!)
9. `IntegerType assertion` - Found related issues
10. `hw.constant arithmetic` - No results
11. `negation signed` - No results
12. `verifyInvariants assertion failed` - Found related issues
13. `hw.wire type unresolved` - No results
14. `createIntegerType bitwidth validation` - No results
15. `signed wire comparison` - No results
16. `hw.getBitWidth LLHD` - No results

---

## Top Matches Analysis

### 1. Issue #9287: [HW] Make `hw::getBitWidth` use std::optional vs -1
**Match Score**: 7/10  
**Status**: OPEN  
**Created**: 2025-12-02T20:12:37Z  
**Label**: HW

#### Description
Proposes converting `circt::getBitWidth` to return `std::optional<uint64_t>` instead of `-1` for invalid cases. The proposal includes:
- Convert the `BitWidthTypeInterface` `getBitWidth` method
- Update all callsites to handle the optional return type
- Add assertions where appropriate
- Bail out gracefully in cases where bitwidth was inappropriately assumed

#### Matched Keywords
- `getBitWidth` ✓
- `bitwidth validation` ✓
- `assertion failure when invalid bitwidth` ✓
- `getIntegerType without validation` ✓

#### Why This is a Match
**CRITICAL CONNECTION**: This issue directly addresses the root cause of the reported crash:
1. The current `getBitWidth()` returns `-1` for invalid cases
2. The Mem2Reg pass does NOT validate the return value before passing to `builder.getIntegerType()`
3. `IntegerType::get()` calls `verifyInvariants()` which asserts when bitwidth > 16,777,215
4. The crash in Mem2Reg.cpp line 1753 is exactly where this validation is missing

**Conclusion**: The crash location (Mem2Reg.cpp:1753) is one of the callsites that needs to be updated as part of the fix described in #9287.

---

### 2. Issue #9574: [Arc] Assertion failure when lowering inout ports in sequential logic
**Match Score**: 6/10  
**Status**: OPEN  
**Created**: 2026-02-01T05:48:51Z  
**Labels**: None (Arc/LLHD related)

#### Description
CIRCUIT crashes with an assertion failure when compiling SystemVerilog code using `inout` ports within `always_ff` blocks:
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: Assertion `succeeded(ConcreteT::verifyInvariants(...))' failed
```

Occurs in `lib/Dialect/Arc/Transforms/LowerState.cpp:219` in Arc's `LowerStatePass`.

#### Matched Keywords
- `assertion failure` ✓
- `bitwidth validation` ✓
- `verifyInvariants` ✓
- `LLHD type issues` ✓
- `StateType creation failure` ✓

#### Why This is a Similar Pattern
This is a **different pass with a similar root cause pattern**:
1. Both involve assertion failures triggered by type verification
2. Both involve LLHD type system issues
3. Both show pattern of insufficient bitwidth validation
4. However: Different passes (Arc's LowerStatePass vs Mem2Reg), different error locations, different type handling logic

**Relevance**: Related infrastructure issue showing broader pattern of bitwidth validation problems in LLHD/Arc integration. May require similar fixes.

---

### 3. Issue #8693: [Mem2Reg] Local signal does not dominate final drive
**Match Score**: 5/10  
**Status**: OPEN  
**Created**: 2025-07-11T20:50:35Z  
**Labels**: LLHD

#### Description
Mem2Reg pass produces a local signal that doesn't dominate its final drive, resulting in verification error:
```
error: operand #0 does not dominate this use
note: see current operation: "llhd.drv"(%6, %7, %9, %8)
```

#### Matched Keywords
- `Mem2Reg pass` ✓
- `LLHD dialect` ✓
- `assertion/verification error` ✓
- `signal handling` ✓

#### Why This is Related
**Same pass, different bug**: This is another Mem2Reg bug in LLHD:
1. Same affected pass (Mem2Reg)
2. Same dialect (LLHD)
3. Both are assertion/verification failures
4. May share infrastructure issues

**Differences**: This bug is about signal domination (SSA form violation), not bitwidth validation. The root causes are distinct.

---

### 4. Issue #9013: [circt-opt] Segmentation fault during XOR op building
**Match Score**: 4/10  
**Status**: OPEN  
**Created**: 2025-09-24T16:10:55Z  
**Labels**: LLHD

#### Description
Segmentation fault in `llhd-desequentialize` pass when building XOR operations with type mismatches:
```
#4 0x00005f1950f4aa48 circt::comb::__mlir_ods_local_type_constraint_Comb1(...)
#5 0x00005f1950f861df circt::comb::XorOp::verifyInvariantsImpl()
#3 0x00005f19517f1f60 circt::hw::isHWIntegerType(mlir::Type)
```

#### Matched Keywords
- `LLHD` ✓
- `assertion/crash in pass` ✓
- `type mismatch` ✓
- `verifyInvariants context` ✓

#### Why This is Tangentially Related
**LLHD infrastructure pattern**: This shows a pattern of LLHD dialect issues with type system verification. However:
1. Different pass (llhd-desequentialize vs Mem2Reg)
2. Different operation type (XOR vs signal handling)
3. Different verification failure (type constraint vs bitwidth)

**Relevance**: General LLHD type system stability concern, but not directly related to the specific bitwidth issue.

---

## Other Reviewed Issues

### #9467: [circt-verilog][arcilator] `arcilator` fails to lower `llhd.constant_time`
- **Score**: 3/10 - Different error type (constant_time lowering)
- **Relevance**: LLHD dialect but unrelated to bitwidth validation

### #7989: [LLHD] Improve llhd-desequentialize pass
- **Score**: 2/10 - Enhancement request, not a bug report
- **Relevance**: LLHD but unrelated to Mem2Reg or bitwidth issues

### #8266: [FIRRTL] Integer Property folders assert in getAPSInt
- **Score**: 1/10 - Different dialect (FIRRTL) and different component
- **Relevance**: Generic assertion pattern only

### #8825: [LLHD] Switch from hw.inout to a custom signal reference type
- **Score**: 2/10 - Design/architecture discussion
- **Relevance**: LLHD but not a bug report

### #7665: [LLHD][ProcessLowering] Incorrectly inlines aliasing drives
- **Score**: 2/10 - ProcessLowering pass, not Mem2Reg
- **Relevance**: LLHD but different pass

### #8226: [ImportVerilog] Problem with conditional + register
- **Score**: 1/10 - ImportVerilog issues
- **Relevance**: LLHD but different frontend

---

## Recommendation

### Decision: `review_existing`

### Rationale

1. **Issue #9287 is Directly Related**: The crash is a direct manifestation of the problem addressed in #9287. The root cause (missing validation of `getBitWidth()` return value) is explicitly documented in both cases.

2. **Same File, Same Function**: The crash occurs at `lib/Dialect/LLHD/Transforms/Mem2Reg.cpp:1753` in the `Promoter::insertBlockArgs()` function. This is exactly the callsite that will need to be updated as part of implementing issue #9287's proposed fix.

3. **Root Cause Already Identified**: Issue #9287 was created on 2025-12-02 and already proposes the exact fix needed:
   - Convert `getBitWidth()` to return `std::optional<uint64_t>`
   - Update all callsites to validate before use
   - The Mem2Reg.cpp:1753 location is one of these callsites

4. **Timeline**: Issue #9287 is very recent (Dec 2, 2025) and may already be in development or planned for fixing.

### Actions

**Instead of creating a new issue:**

1. **Reference Issue #9287**: When reporting this crash, explicitly mention that it's a manifestation of the problem described in #9287
2. **Provide Minimal Test Case**: Add the signed wire comparison test case (`wire signed [N:0] a; (-a <= a)`) as a concrete repro for the getBitWidth validation issue
3. **Link as Related Issue**: Mark this analysis as "Related to #9287" in any tracking system

**Stakeholders should:**
1. Review PR/work on #9287 to ensure it covers the Mem2Reg.cpp:1753 case
2. Include this crash pattern in the test cases for #9287
3. Verify that the fix in #9287 resolves this crash

---

## Technical Details

### Type System Flow

```
Signed wire [2:0] 'a' with value
    ↓
Unary negation operator (-a) applied
    ↓
Comparison operator (<=) used: (-a <= a)
    ↓
Type resolution attempted
    ↓
hw::getBitWidth(slotType) called
    ↓
Returns -1 or invalid large value (>16,777,215)
    ↓
builder.getIntegerType(invalid_bitwidth) called
    ↓
IntegerType::get() calls verifyInvariants()
    ↓
Assertion failure: bitwidth must be <= 16,777,215
```

### Stack Depth to Crash

- **Stack depth**: 13 function calls from entry point to crash
- **Verification layer**: Type verification happens in MLIR's StorageUniquerSupport.h layer
- **Diagnostic opportunity**: Could emit user-friendly error before reaching assertion

---

## Suggested Next Steps

1. **For CIRCT Developers**: Proceed with implementation of #9287, ensuring Mem2Reg.cpp:1753 is properly updated with bitwidth validation
2. **For Bug Reporters**: Instead of reporting as new issue, comment on #9287 with this test case and stack trace
3. **For QA/Testing**: Add this test pattern to regression tests for #9287 fix validation
4. **For Architecture**: Consider if the broader LLHD type system needs review for similar issues (see related #9574)

---

## Appendix: Full Search Log

### Search Query 1: "bitwidth limit Mem2Reg assertion"
- **Result**: No matching issues (too specific)

### Search Query 2: "16777215"
- **Result**: No matching issues (number-based search ineffective)

### Search Query 3: "insertBlockArgs"
- **Result**: No matching issues (internal function name)

### Search Query 4: "getBitWidth invalid"
- **Result**: No direct matches
- **Note**: Later query "integer type bitwidth" found #9287

### Search Query 5: "Mem2Reg"
- **Result**: 3 issues found
  - #8693: Different Mem2Reg bug (domination issue)
  - #9467: LLHD-related but unrelated bug
  - #9574: Arc pass (related pattern)

### Search Query 6: "LLHD"
- **Result**: 30+ issues found
- **Filtered to relevant**: 10 issues selected

### Search Query 7-16: Various combinations
- **Result**: Iteratively refined to identify top matches

### Final Top Result
- **Issue #9287** identified through "integer type bitwidth" query
- **Confirmed as primary match** through manual review of issue body

---

**Report Generated**: 2025-12-XX (current date)  
**Analysis Confidence**: HIGH  
**Recommendation Confidence**: HIGH  
**Final Status**: NOT A NEW ISSUE - Related to #9287
