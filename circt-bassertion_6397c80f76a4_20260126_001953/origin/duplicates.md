# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| **Recommendation** | `likely_new` |
| **Top Score** | 7.5 / 15.0 |
| **Top Issue** | #8930 |
| **Exact Duplicate** | No |
| **Related Issues** | 6 |

## Search Queries

1. `string port MooreToCore`
2. `dyn_cast non-existent value`
3. `moore string type`
4. `DynamicStringType`
5. `sanitizeInOut`
6. `string module port`
7. `getModulePortInfo`

## Target Bug Characteristics

- **Dialect**: Moore
- **Failing Pass**: MooreToCore
- **Assertion**: `dyn_cast on a non-existent value`
- **Crash Location**: `MooreToCore.cpp:259` (getModulePortInfo)
- **Key Feature**: String type output port causes crash

## Search Results (Sorted by Similarity Score)

### 1. [#8930](https://github.com/llvm/circt/issues/8930) - [MooreToCore] Crash with sqrt/floor
| State | Score | Labels |
|-------|-------|--------|
| open | **7.5** | Moore |

**Match Reasons:**
- Title contains 'MooreToCore' (+2.0)
- Body contains 'dyn_cast on a non-existent value' (+3.0)
- Dialect label 'Moore' matches (+1.5)
- Body contains 'MooreToCore' (+1.0)

**Analysis:** Same assertion message but different crash cause. Issue #8930 crashes on sqrt/floor operations with real type conversion, while our bug crashes on string port type handling. **Not a duplicate.**

---

### 2. [#8332](https://github.com/llvm/circt/issues/8332) - [MooreToCore] Support for StringType from moore to llvm dialect
| State | Score | Labels |
|-------|-------|--------|
| open | **7.0** | - |

**Match Reasons:**
- Title contains 'MooreToCore' (+2.0)
- Title contains 'StringType' (+2.0)
- Body mentions related string/sim concepts (+3.0)

**Analysis:** This is a **feature request** for StringType support in Moore→LLVM lowering. Our bug is a specific crash when string types appear in module ports. Related but not a duplicate - our crash shows a missing validation case. **Not a duplicate but highly related.**

---

### 3. [#8219](https://github.com/llvm/circt/issues/8219) - [ESI] Assertion: dyn_cast on a non-existent value
| State | Score | Labels |
|-------|-------|--------|
| closed | **6.0** | ESI |

**Match Reasons:**
- Title contains 'dyn_cast' (+2.0)
- Title contains 'assertion' (+2.0)
- Body contains same assertion message (+3.0)

**Analysis:** Same assertion message but different dialect (ESI) and completely different code path. The common assertion is a generic LLVM casting error that can occur in many places. **Not a duplicate.**

---

### 4. [#9315](https://github.com/llvm/circt/issues/9315) - [FIRRTL] ModuleInliner removes NLA referred by circt.tracker
| State | Score | Labels |
|-------|-------|--------|
| closed | **4.0** | - |

**Analysis:** FIRRTL dialect, completely unrelated. **Not a duplicate.**

---

### 5. [#8173](https://github.com/llvm/circt/issues/8173) - [ImportVerilog] Crash on ordering-methods-reverse test
| State | Score | Labels |
|-------|-------|--------|
| open | **3.0** | bug, ImportVerilog |

**Analysis:** Related to string handling (string arrays) but different crash mechanism - this crashes on casting string to simple bit vector. **Not a duplicate but related to string support issues.**

---

### 6. [#7471](https://github.com/llvm/circt/issues/7471) - Crash in circt-verilog: Assertion succeeded(result)
| State | Score | Labels |
|-------|-------|--------|
| closed | **2.0** | - |

**Analysis:** Different assertion message, unrelated crash. **Not a duplicate.**

---

## Conclusion

**Recommendation: `likely_new`**

No exact duplicate found. The closest match (#8930) shares the same assertion message but has a completely different trigger (sqrt/floor operations vs string ports).

**Key Observations:**
1. Issue #8332 indicates StringType support in MooreToCore is still being developed
2. Our crash reveals a specific case: string type module ports are not properly validated
3. The assertion `dyn_cast on a non-existent value` is a common LLVM error that appears in multiple unrelated issues

**Recommendation for Filing:**
- Create a new issue
- Reference #8332 as related work on StringType support
- Clearly distinguish from #8930 (different crash trigger)
- Suggest fix direction: validate port types before sanitizeInOut()
