# CIRCT Bug Duplicate Check Report

**Date**: 2026-01-31T22:19:46.915175  
**Testcase ID**: 260128-00000ac8  

## Executive Summary

| Metric | Value |
|--------|-------|
| **Recommendation** | `review_existing` |
| **Top Similarity Score** | 12.0/20 |
| **Most Similar Issue** | #9260 |
| **Total Results Found** | 24 |

### Analysis Result
Found moderately similar issue (score >= 10)

---

## Search Methodology

### Keywords Extracted
- **Tool**: arcilator
- **Pass**: InferStateProperties
- **Crash Type**: assertion
- **Dialect**: arc
- **Function**: applyEnableTransformation
- **Error Pattern**: cast<IntegerType>
- **Type Issue**: packed struct

### Search Queries Used
- `arcilator crash`
- `InferStateProperties assertion`
- `packed struct`
- `cast<IntegerType>`
- `struct array`

### Similarity Scoring System
| Score | Classification | Criteria |
|-------|-----------------|----------|
| 20 | Perfect Match | Same function, same error type |
| 15 | High Relevance | Same pass, different error |
| 10 | Moderate | Same dialect, related operations |
| 5 | Weak | Same error type or tool |
| 0 | No Match | No matching keywords |

---

## Top 5 Most Similar Issues

### 1. Issue #9260
**Title**: Arcilator crashes in Upload Release Artifacts CI  
**State**: OPEN  
**Similarity Score**: 12.0/20  
**Matched Queries**: `arcilator crash`  
**Labels**: `bug`, `Arc`  

### 2. Issue #6373
**Title**: [Arc] Support hw.wires of aggregate types  
**State**: OPEN  
**Similarity Score**: 10.0/20  
**Matched Queries**: `struct array`  
**Labels**: `Arc`  

### 3. Issue #8065
**Title**: [LLHD][Arc] Indexing and slicing lowering from Verilog to LLVM IR  
**State**: OPEN  
**Similarity Score**: 10.0/20  
**Matched Queries**: `struct array`  
**Labels**: None  

### 4. Issue #8930
**Title**: [MooreToCore] Crash with sqrt/floor  
**State**: OPEN  
**Similarity Score**: 7.0/20  
**Matched Queries**: `cast<IntegerType>`  
**Labels**: `Moore`  

### 5. Issue #3289
**Title**: [PyCDE] ConcatOp of arrays causes crash  
**State**: OPEN  
**Similarity Score**: 7.0/20  
**Matched Queries**: `cast<IntegerType>`  
**Labels**: `PyCDE`  

---

## Detailed Results

All 24 results found are listed below (sorted by similarity):

- **#9260** (12.0pts) - Arcilator crashes in Upload Release Artifacts CI (OPEN)
- **#6373** (10.0pts) - [Arc] Support hw.wires of aggregate types (OPEN)
- **#8065** (10.0pts) - [LLHD][Arc] Indexing and slicing lowering from Verilog to LLVM IR (OPEN)
- **#8930** (7.0pts) - [MooreToCore] Crash with sqrt/floor (OPEN)
- **#3289** (7.0pts) - [PyCDE] ConcatOp of arrays causes crash (OPEN)
- **#8292** (7.0pts) - [MooreToCore] Support for Unsized Array Type (OPEN)
- **#3853** (5.0pts) - [ExportVerilog] Try to make bind change the generated RTL as little as possible (OPEN)
- **#5138** (3.0pts) - [ExportVerilog] `disallowPackedStructAssignments` also needs to consider `hw.aggregate_constant` (OPEN)
- **#9076** (3.0pts) - [FIRRTL] Preserve aggregate of memory data type to make LEC friendly (OPEN)
- **#2439** (0.0pts) - [ExportVerilog] Remove temporary for aggregate outputs (OPEN)
- **#6614** (0.0pts) - [LowerSeqToSV] FIRRTL enum with clock field leads to invalid IR (OPEN)
- **#2504** (0.0pts) - [ExportVerilog] Incorrect verilog output for bitcast + zero width aggregate types (OPEN)
- **#2567** (0.0pts) - [HW][FIRRTL] Endian difference between struct and array (OPEN)
- **#2329** (0.0pts) - [LowerToHW] Use type decl for Bundle type lowering (OPEN)
- **#7535** (0.0pts) - [MooreToCore] VariableOp lowered failed (OPEN)
- **#8476** (0.0pts) - [MooreToCore] Lower exponentiation to `math.ipowi` (OPEN)
- **#6271** (0.0pts) - [HW] Unequal equal types on instancing a module outputting a parametrized array / ArrayType sizeAttr defaults to 64bit (OPEN)
- **#6816** (0.0pts) - [HGLDD] Emit HW struct and array types (OPEN)
- **#6983** (0.0pts) -  Elaborate chisel type annotation from firtool to generate debug information for the Tywaves project (OPEN)
- **#2590** (0.0pts) - [ExportVerilog] Unnecessary temporary wire for bitcast between array and integer (OPEN)
- **#391** (0.0pts) - Cache non-inout bits in composite RTL types (OPEN)
- **#1352** (0.0pts) - [FIRRTL] Add create vector/bundle ops (OPEN)
- **#2419** (0.0pts) - [HW][SV] Use FieldID in HW/SV dialects (OPEN)
- **#5253** (0.0pts) - [CI][FIRRTL][LLHD] valgrind failures on nightly (GCC, asserts=ON, shared=OFF) (OPEN)


---

## Recommendation Details

**Decision**: `REVIEW_EXISTING`


**Reason**: The top similar issue (#9260) has a similarity score of 12.0/20, indicating potential overlap.

**Action Items**:
1. Review issue #9260: Arcilator crashes in Upload Release Artifacts CI
2. Compare error signatures and stack traces
3. Determine if this is a duplicate or a related but distinct issue
4. Check if the issue is already in progress or fixed in latest main branch

**Next Steps**:
- If confirmed duplicate: Link issues and close the new one
- If related but distinct: Document the relationship and proceed with separate fix
- If unrelated: Proceed with new issue report


---

## Metadata
- **Search Timestamp**: 2026-01-31T22:19:32.042568
- **Total Issues Analyzed**: 24
- **Queries Executed**: 5

Generated by: check-duplicates-worker
