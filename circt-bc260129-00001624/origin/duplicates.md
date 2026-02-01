# Duplicate Check Report

**Generated:** 2026-02-01T10:47:09.238425

## Test Case Information
- **Testcase ID:** 260129-00001624
- **Dialect:** MOORE
- **Pass:** mooretocore
- **Crash Type:** assertion
- **Keywords:** union packed, output port, typedef

## Crash Signature
- **Assertion Message:** dyn_cast on a non-existent value
- **Missing Types:** moore::uniontype, moore::unpackeduniontype

## Search Results
- **Total Issues Searched:** 13
- **Scoring Algorithm:** keyword(1) + dialect(2) + pass(2) + assertion(3) + missing_type(3) + inout(3) + union(2)

## Top Matches

### Issue #8930 - Highest Similarity Score: 7

**Title:** [MooreToCore] Crash with sqrt/floor

**Status:** OPEN

**Author:** uenoku

**Created:** 2025-09-06T09:21:38Z

**Matches:**
- dialect:moore
- pass:mooretocore
- assertion:dyn_cast

**Body Preview:**
```
It failed when trying to compile https://github.com/pulp-platform/ELAU/blob/b0d113aff6a2d800076f5ebb84f09fba93625bc7/src/SqrtArrUns.sv#L96-L105. 
```
moore.module @behavioural_SqrtArrUns(in %X : !moore.l8, out Q : !moore.l4, out R : !moore.l4) {
  %0 = moore.constant 2 : l8
  %1 = moore.conversion %X : !moore.l8 -> !moore.real
  %2 = moore.builtin.sqrt %1 : real
  %3 = moore.builtin.floor %2 : real
  %4 = moore.conversion %3 : !moore.real -> !moore.l4
  %5 = moore.zext %4 : l4 -> l8
  %6 = moore...
```

**Link:** https://github.com/llvm/circt/issues/8930

---

## All Top 10 Similar Issues

| Rank | Issue # | Score | Title | Matches |
|------|---------|-------|-------|---------|
| 1 | #8930 | 7 | [MooreToCore] Crash with sqrt/floor... | dialect:moore, pass:mooretocore, assertion:dyn_cast |
| 2 | #7629 | 4 | [MooreToCore] Support net op... | dialect:moore, pass:mooretocore |
| 3 | #8163 | 4 | [MooreToCore] Out-of-bounds moore.extract lowered ... | dialect:moore, pass:mooretocore |
| 4 | #8176 | 4 | [MooreToCore] Crash when getting values to observe... | dialect:moore, pass:mooretocore |
| 5 | #8211 | 4 | [MooreToCore]Unexpected observed values in llhd.wa... | dialect:moore, pass:mooretocore |
| 6 | #8215 | 4 | [MooreToCore] OOB array slices of unpacked arrays ... | dialect:moore, pass:mooretocore |
| 7 | #8269 | 4 | [MooreToCore] Support `real` constants... | dialect:moore, pass:mooretocore |
| 8 | #8476 | 4 | [MooreToCore] Lower exponentiation to `math.ipowi`... | dialect:moore, pass:mooretocore |
| 9 | #8973 | 4 | [MooreToCore] Lowering to math.ipow?... | dialect:moore, pass:mooretocore |
| 10 | #7531 | 2 | [Moore] Input triggers assertion in canonicalizer ... | dialect:moore |

## Recommendation

- **Action:** **likely_duplicate**
- **Confidence:** HIGH
- **Highest Similarity Score:** 7
- **Most Similar Issue:** #8930

## Summary

This test case appears to be related to **Issue #8930**. 

Both issues:
1. Involve the **Moore dialect** and **MooreToCore pass**
2. Trigger **dyn_cast assertions** on null types
3. Represent missing type converter support in CIRCT

**Recommended Action:** Before creating a new issue, review Issue #8930 to determine if:
- It is the same root cause (different trigger construct)
- It should be merged into that issue
- Or if a new issue is needed with different characteristics

**Similarity Metrics:**
- dialect:moore
- pass:mooretocore
- assertion:dyn_cast

## Search Methodology

1. Extracted key information from test case:
   - Dialect: moore
   - Pass: mooretocore
   - Keywords: union packed, output port, typedef
   
2. Searched llvm/circt repository for:
   - Issues with 'moore' label
   - Issues mentioning keywords, pass name, assertion message
   - Issues related to missing type conversions
   
3. Scored results using weighted matching:
   - Keywords: weight 1 per match
   - Dialect: weight 2
   - Pass name: weight 2
   - Assertion message: weight 3
   - Missing type: weight 3
   - InOutType feature: weight 3
   - Union feature: weight 2

---
**Report Status:** COMPLETED
**Date:** 2026-02-01 10:47:09
