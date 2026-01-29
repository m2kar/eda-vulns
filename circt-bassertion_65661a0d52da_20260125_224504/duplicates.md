# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| **Recommendation** | `likely_new` |
| **Top Score** | 14.5 |
| **Most Similar Issue** | [#8930](https://github.com/llvm/circt/issues/8930) |
| **Issues Analyzed** | 5 |

## Current Bug Signature

- **Dialect**: Moore
- **Failing Pass**: MooreToCorePass
- **Crash Type**: Assertion failure
- **Assertion**: `dyn_cast on a non-existent value`
- **Crash Location**: `include/circt/Dialect/HW/PortImplementation.h:sanitizeInOut:177`
- **Trigger Location**: `lib/Conversion/MooreToCore/MooreToCore.cpp:getModulePortInfo:259`
- **Root Cause**: String type used as module port, type conversion fails/returns null

## Similar Issues

### 1. [#8930](https://github.com/llvm/circt/issues/8930) - [MooreToCore] Crash with sqrt/floor ⚠️ HIGH SIMILARITY

| Field | Value |
|-------|-------|
| **Score** | 14.5 |
| **State** | Open |
| **Labels** | Moore |

**Score Breakdown:**
- Title keywords (MooreToCore, Crash): 4.0
- Assertion match (`dyn_cast on a non-existent value`): 3.0
- Dialect match (Moore): 1.5
- Pass match (MooreToCore): 2.0
- Body keywords: 4.0

**Analysis:**
Very similar crash in MooreToCore with the **same assertion message**. However, the triggers are different:
- **#8930**: `ConversionOp` handling `real` type, fails in `hw::getBitWidth()`
- **Current**: `SVModuleOp` with `string` port type, fails in `sanitizeInOut()`

Both stem from incomplete type handling in MooreToCore but for **different unsupported types**.

---

### 2. [#8176](https://github.com/llvm/circt/issues/8176) - [MooreToCore] Crash when getting values to observe

| Field | Value |
|-------|-------|
| **Score** | 8.5 |
| **State** | Open |
| **Labels** | Moore |

**Analysis:**
MooreToCore crash but different mechanism - unattached region issue, not type conversion failure.

---

### 3. [#7627](https://github.com/llvm/circt/issues/7627) - [MooreToCore] Unpacked array causes crash

| Field | Value |
|-------|-------|
| **Score** | 8.5 |
| **State** | Closed |
| **Labels** | bug, Moore |

**Analysis:**
Already fixed. Different trigger (array extraction vs string port type).

---

### 4. [#8219](https://github.com/llvm/circt/issues/8219) - [ESI] Assertion: dyn_cast on a non-existent value

| Field | Value |
|-------|-------|
| **Score** | 8.0 |
| **State** | Closed |
| **Labels** | ESI |

**Analysis:**
Same assertion message but in ESI dialect, completely different code path.

---

### 5. [#9315](https://github.com/llvm/circt/issues/9315) - [FIRRTL] ModuleInliner removes NLA

| Field | Value |
|-------|-------|
| **Score** | 6.0 |
| **State** | Closed |
| **Labels** | - |

**Analysis:**
Same assertion message but in FIRRTL dialect. Unrelated root cause.

---

## Recommendation

### Decision: `likely_new` (Create new issue, but reference related issues)

**Reasoning:**

1. **Same assertion pattern** as #8930 (score 14.5 > 8.0 threshold)
2. **Different root cause**: 
   - #8930: `real` type in `ConversionOp` → fails in `getBitWidth()`
   - Current: `string` type as module port → fails in `sanitizeInOut()`
3. Both issues indicate **incomplete type handling in MooreToCore**
4. The fix for #8930 (if any) may not cover this case

### Suggested Action:

Create a **new issue** with:
- Title: `[MooreToCore] Crash with string type as module port`
- Reference #8930 as related issue
- Note: Similar pattern, different unsupported type

## Search Queries Used

1. `string type port` - 0 results
2. `MooreToCore crash` - 3 results
3. `dyn_cast assertion` - 3 results
4. `Moore` - 10 results
5. `getModulePortInfo` - 0 results
6. `sanitizeInOut` - 0 results
7. `InOutType port` - 0 results
