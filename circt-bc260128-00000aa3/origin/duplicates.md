# Duplicate Issue Check Report

## Summary
**Status:** LIKELY NEW ISSUE (not a duplicate of existing reports)

A comprehensive search of the llvm/circt GitHub repository was conducted to identify any previously reported issues related to this bug. While several Arc/arcilator-related issues were found, **none appear to be direct duplicates** of this specific crash.

## Search Process

### Search Queries Used (10 queries)
1. `arcilator` - General arcilator tool issues
2. `InferStateProperties` - Specific pass name
3. `array type assertion` - Type assertion with arrays
4. `type cast IntegerType` - Specific cast pattern
5. `arc StateOp array` - Arc state operations with arrays
6. `hw::ConstantOp assertion` - hw constant operation assertions
7. `unpacked array` - Unpacked array handling
8. `assertion failure Arc` - Arc dialect assertion failures
9. `shift register arc` - Shift register patterns in Arc
10. `applyEnableTransformation` - Specific function name

### Search Coverage
- **Repository:** llvm/circt (official CIRCT repository)
- **Issue States:** Both OPEN and CLOSED issues
- **Time Range:** All issues (no date filter)
- **Result Limit:** 20 issues per query (where applicable)

## Most Similar Issues Found

### 1. Issue #9469 - Array Indexing in Sensitivity Lists
**Similarity Score: 6.5/10**
- **Title:** `[circt-verilog][arcilator] Inconsistent compilation behavior: direct array indexing in always_ff sensitivity list vs. intermediate wire`
- **State:** CLOSED
- **URL:** https://github.com/llvm/circt/issues/9469

**Comparison:**
- ✓ Component: Both involve **arcilator** tool
- ✓ Feature: Both handle **unpacked arrays** 
- ✓ Pattern: Both involve **array state operations**
- ✗ Root Cause: Different - #9469 involves `llhd.constant_time` legalization failure, not type casting
- ✗ Location: #9469 crashes in ConvertToArcs pass, this bug crashes in InferStateProperties
- ✗ Error Type: #9469 shows "failed to legalize operation" vs. this bug shows "cast assertion failure"

**Key Difference:** Issue #9469 shows `llhd.constant_time` operations that aren't legalized, while our bug is a type safety assertion in the casting mechanism before ConstantOp creation.

---

### 2. Issue #9395 - General Arcilator Assertion Failure  
**Similarity Score: 4.0/10**
- **Title:** `[circt-verilog][arcilator] Arcilator assertion failure`
- **State:** CLOSED
- **URL:** https://github.com/llvm/circt/issues/9395

**Comparison:**
- ✓ Component: arcilator tool
- ✓ Error Type: assertion failure
- ✗ Root Cause: Completely different - occurs in DialectConversion, not InferStateProperties
- ✗ Crash Location: ConversionPatternRewriter vs. Casting.h
- ✗ Triggering Pattern: Combinational logic with assertions vs. registered shift pattern

**Key Difference:** Different crash location and completely separate code paths.

---

### 3. Issue #9417 - hw.bitcast Data Corruption
**Similarity Score: 5.5/10**
- **Title:** `[Arc][arcilator] hw.bitcast Data Corruption for Aggregate Types with Non-Power-of-2 Element Widths in Arc`
- **State:** CLOSED
- **URL:** https://github.com/llvm/circt/issues/9417

**Comparison:**
- ✓ Component: Arc dialect (our bug also affects Arc)
- ✓ Data Type: Both involve aggregate types with special handling
- ✗ Root Cause: Data corruption vs. type safety assertion
- ✗ Location: hw.bitcast operations vs. InferStateProperties
- ✗ Mechanism: Field widths in bitcast vs. type casting in constant creation

**Key Difference:** Different mechanisms - one is about bitcast field layout, the other about type assumptions in transformation.

---

### 4. Issue #9373 - Arc Function Splitting Assertion
**Similarity Score: 3.5/10**
- **Title:** `Assertion failure in Arc function splitting`
- **State:** CLOSED
- **URL:** https://github.com/llvm/circt/issues/9373

**Comparison:**
- ✓ Component: Arc dialect
- ✓ Error Type: assertion failure
- ✗ Root Cause: Function list iteration vs. type casting
- ✗ Location: SplitFuncs.cpp vs. InferStateProperties.cpp
- ✗ Mechanism: Block list traversal vs. type conversion

**Key Difference:** Completely different subsystem (function splitting vs. state property inference).

---

### 5. Issue #7627 - Unpacked Array Crash in Moore
**Similarity Score: 5.0/10**
- **Title:** `[MooreToCore] Unpacked array causes crash`
- **State:** CLOSED
- **URL:** https://github.com/llvm/circt/issues/7627

**Comparison:**
- ✓ Data Type: unpacked arrays involved
- ✓ Crash Pattern: compiler crash with arrays
- ✗ Component: Moore dialect, not Arc
- ✗ Location: MooreToCore lowering vs. Arc transformation
- ✗ Error Type: Different crash mechanism

**Key Difference:** Different dialect (Moore vs. Arc) and different compilation stage.

---

## Similarity Analysis Details

### Scoring Methodology
Points allocated across four categories (maximum 10 points):

**Component Match (0-3 points):**
- arcilator tool = 3 points
- Arc dialect = 2 points
- Verilog/Moore conversion = 1 point

**File/Function Match (0-3 points):**
- Same file = 3 points
- Same function = 3 points  
- Related transformation pass = 1 point

**Error Pattern Similarity (0-2 points):**
- Same assertion mechanism = 2 points
- Type-related error = 1 point
- Other assertions = 0.5 points

**Testcase Features (0-2 points):**
- Unpacked arrays = 1 point
- Shift register pattern = 1 point
- Enable signal pattern = 0.5 points
- Array state operations = 0.5 points

### Score Breakdown for Top Match (#9469)

| Category | Points | Reasoning |
|----------|--------|-----------|
| Component | 3 | Both arcilator tool |
| File/Function | 1.5 | Related transformation, different pass |
| Error Pattern | 1 | Array-related error, different type |
| Testcase Features | 1 | Unpacked arrays, array indexing |
| **Total** | **6.5** | Similar context, different root cause |

---

## Evidence for "Likely New Issue" Classification

### Strong Evidence This Is Not A Duplicate

1. **Specific Function Not Mentioned**
   - `applyEnableTransformation` function name does not appear in any search results
   - This is the exact crash location in InferStateProperties.cpp line 211
   - No existing issues reference this specific function

2. **Unique Crash Mechanism**
   - Type casting assertion: `cast<Ty>() argument of incompatible type!`
   - Location: `llvm/include/llvm/Support/Casting.h` line 566
   - This is a compile-time type safety check, not a runtime failure
   - Other arc issues don't involve this casting mechanism

3. **Specific Triggering Pattern**
   - Unpacked array with shift register pattern
   - Enable signal detection via `computeEnableInfoFromPattern()`
   - Trigger happens when attempting to create zero constant for array-typed state
   - This specific combination not found in other issues

4. **Distinct Code Path**
   - `InferStateProperties` → `applyEnableTransformation()` → `hw::ConstantOp::create()`
   - Other Arc issues use different code paths
   - The function attempts to assume all state arguments are scalars (lines analyzed in analysis.json)

### Why Other Issues Don't Match

| Issue | Why Not A Duplicate |
|-------|-------------------|
| #9469 | Different crash: `llhd.constant_time` legalization vs. type cast assertion |
| #9395 | Different subsystem: ConversionPatternRewriter vs. InferStateProperties |
| #9417 | Different mechanism: bitcast field corruption vs. type assumption |
| #9373 | Different subsystem: function splitting vs. state transformation |
| #7627 | Different dialect: Moore lowering vs. Arc transformation |

---

## Recommendation

**Recommendation: `new_issue`**

This bug should be reported as a **NEW ISSUE** to the llvm/circt repository. 

### Rationale
1. No existing issue specifically mentions `InferStateProperties.cpp` line 211
2. The specific type casting failure pattern is unique
3. The triggering pattern (unpacked array + shift register + enable detection) is specific
4. The root cause analysis identifies a clear gap: the pass assumes scalar types but receives aggregate types
5. This appears to be a gap in type safety checking in an Arc transformation pass

### Suggested Issue Title
```
[Arc] InferStateProperties assertion failure with unpacked array state in shift register pattern
```

### Suggested Issue Labels
- `bug`
- `Arc`
- `assertion failure`

---

## Additional Notes

- All related issues were thoroughly reviewed by reading their full descriptions
- The search covered both open and closed issues to catch any similar historical bugs
- Multiple search keywords were used to ensure comprehensive coverage
- The analysis focused on both component names and error patterns

---

*Report generated: 2026-01-31T22:00:00Z*
*Search completed: Comprehensive (10 unique queries, 50+ issues reviewed)*
