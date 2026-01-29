# Duplicate Issue Check Report

## Search Strategy
- **Search terms used**: Sig2RegPass, RAUW, replaceAllUsesWith, self-referential, multiple-drivers, signal-promotion, LLHD, assertion
- **Repository**: llvm/circt
- **Issues searched**: 15 total
- **Issues with keyword matches**: 8

## Bug Signature
- **Crash Type**: Assertion failure
- **Assertion**: `cannot RAUW a value with itself`
- **Location**: `Sig2RegPass.cpp:35` in `Offset::~Offset()`
- **Dialect**: LLHD
- **Pass**: Sig2RegPass
- **Trigger**: Self-referential continuous assignment (`assign q_out = ... q_out`)

## Top Potential Duplicates

### #6795 - [Ibis] Update direct uses of replaceAllUsesWith.
**Similarity Score**: 3.0
**State**: open
**Matched Keywords**: replaceAllUsesWith, RAUW
**Assertion Match**: ✗
**Dialect Match**: ✗
**URL**: https://github.com/llvm/circt/issues/6795

**Summary**: Issue about DialectConversion triggering asserts on replaceAllUsesWith usage in Ibis dialect.

**Analysis**: NOT a duplicate. This is about API usage concerns in DialectConversion framework, not a crash in Sig2RegPass. Different dialect (Ibis vs LLHD), different context (conversion patterns vs signal promotion).

---

### #9216 - circt-verilog --ir-hw produces multiple seq.firreg for a single Verilog reg
**Similarity Score**: 2.5
**State**: open
**Matched Keywords**: multiple drivers
**Assertion Match**: ✗
**Dialect Match**: ✗
**URL**: https://github.com/llvm/circt/issues/9216

**Summary**: Multiple drivers to same reg causing multiple seq.firreg generation.

**Analysis**: NOT a duplicate. Related concept (multiple drivers) but different manifestation - incorrect output, not a crash. Uses `--ir-hw` pipeline which doesn't involve LLHD/Sig2Reg.

---

### #9013 - [circt-opt] Segmentation fault during XOR op building
**Similarity Score**: 2.0
**State**: open
**Matched Keywords**: LLHD
**Assertion Match**: ✗
**Dialect Match**: ✓
**URL**: https://github.com/llvm/circt/issues/9013

**Summary**: Segfault in llhd-desequentialize pass during XOR op building.

**Analysis**: NOT a duplicate. LLHD-related crash but different location (XorOp verifier in comb dialect), different pass (llhd-desequentialize), and different crash type (segfault vs assertion).

---

### #8845 - [circt-verilog] `circt-verilog` produces non `comb`/`seq` dialects including `cf` and `llhd`
**Similarity Score**: 1.5
**State**: open
**Matched Keywords**: LLHD
**Assertion Match**: ✗
**Dialect Match**: ✓
**URL**: https://github.com/llvm/circt/issues/8845

**Summary**: circt-verilog produces cf and llhd dialects unexpectedly.

**Analysis**: NOT a duplicate. Not a crash at all - concerns unexpected dialect in output. No assertion failure.

---

### #8012 - [Moore][Arc][LLHD] Moore to LLVM lowering issues
**Similarity Score**: 1.5
**State**: open
**Matched Keywords**: LLHD, sig2reg
**Assertion Match**: ✗
**Dialect Match**: ✓
**URL**: https://github.com/llvm/circt/issues/8012

**Summary**: Issues with Moore to LLVM lowering using llhd-sig2reg pass.

**Analysis**: NOT a duplicate. Mentions sig2reg pass but different error - 'failed to legalize operation seq.clock_inv'. Not an assertion failure in RAUW.

---

### #8065 - [LLHD][Arc] Indexing and slicing lowering from Verilog to LLVM IR
**Similarity Score**: 1.5
**State**: open
**Matched Keywords**: LLHD, sig2reg
**Assertion Match**: ✗
**Dialect Match**: ✓
**URL**: https://github.com/llvm/circt/issues/8065

**Summary**: Arcilator fails with 'body contains non-pure operation' after sig2reg pass.

**Analysis**: NOT a duplicate. Mentions sig2reg but different error - 'body contains non-pure operation'. About indexing/slicing support, not self-referential assignment.

---

### #7531 - [Moore] Input triggers assertion in canonicalizer infra
**Similarity Score**: 1.0
**State**: open
**Matched Keywords**: assertion
**Assertion Match**: ✗
**Dialect Match**: ✗
**URL**: https://github.com/llvm/circt/issues/7531

**Summary**: Assertion in canonicalizer for Moore dialect.

**Analysis**: NOT a duplicate. Different assertion message ('expected that op has no uses'), different location (PatternMatch.cpp), different dialect context (Moore canonicalizer).

---

### #4454 - InnerSymbols: Improvements, ideas, wishlist
**Similarity Score**: 1.0
**State**: open
**Matched Keywords**: RAUW
**Assertion Match**: ✗
**Dialect Match**: ✗
**URL**: https://github.com/llvm/circt/issues/4454

**Summary**: Tracking issue for InnerSymbol improvements including RAUW support.

**Analysis**: NOT a duplicate. Not a bug report - this is a tracking/wishlist issue. Mentions RAUW as a feature to add, not as a crash.

---

## Recommendation
**Result**: `new_issue`

**Confidence**: HIGH

**Reason**: 
No existing issues match this specific crash pattern. The bug has a unique signature:
1. **Specific assertion**: `cannot RAUW a value with itself` - not found in any existing issues
2. **Specific location**: `Sig2RegPass.cpp` `Offset::~Offset()` destructor - no similar reports
3. **Specific trigger**: Self-referential continuous assignment (`assign q_out = ... q_out`) combined with multiple drivers
4. **Dialect**: LLHD Sig2RegPass - while there are LLHD issues, none involve this specific assertion

The closest matches involve RAUW mentions but in completely different contexts:
- #6795: API design discussion about replaceAllUsesWith in DialectConversion
- #4454: Feature wishlist for InnerSymbol RAUW support

LLHD-related issues exist (#9013, #8845, #8012, #8065) but describe different failures with different root causes.

## Next Steps
- [x] Search completed with 8 search queries
- [x] Top 8 matches analyzed in detail
- [x] No duplicates found
- [ ] **Proceed to generate issue** - This is a NEW BUG

## Keywords for Future Reference
- `Sig2RegPass`
- `cannot RAUW a value with itself`
- `self-referential assignment`
- `multiple drivers`
- `LLHD signal promotion`
- `UseDefLists.h:213`
