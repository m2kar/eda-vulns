# Duplicate Check Report

## Summary

| Field | Value |
|-------|-------|
| **Recommendation** | `new_issue` |
| **Top Score** | 55/100 |
| **Top Issue** | #8930 |
| **Duplicate Threshold** | 70 |

## Search Strategy

### Keywords Used
- `string`, `port`, `DynamicStringType`, `InOutType`
- `MooreToCore`, `sanitizeInOut`, `dyn_cast`
- `assertion`, `type conversion`

### Search Queries Executed
1. `string type MooreToCore port`
2. `sanitizeInOut assertion`
3. `DynamicStringType`
4. `Moore string port`
5. `dyn_cast assertion`
6. `getModulePortInfo`
7. `MooreToCore crash`
8. `string port assertion`
9. `InOutType sim dialect`
10. `PortImplementation.h`

## Candidate Issues

### #8930 - [MooreToCore] Crash with sqrt/floor (Score: 55)
- **State**: OPEN
- **Labels**: Moore
- **URL**: https://github.com/llvm/circt/issues/8930
- **Match Reason**: Same dialect (Moore), same pass (MooreToCore), dyn_cast assertion failure, but different crash location (ConversionOp vs getModulePortInfo)

**Score Breakdown**:
| Category | Score |
|----------|-------|
| Dialect Match | 15 |
| Pass Match | 15 |
| Crash Type Match | 10 |
| Keyword Match | 15 |
| Assertion Pattern | 0 |

---

### #8382 - [FIRRTL] Crash with fstring type on port (Score: 45)
- **State**: OPEN
- **Labels**: -
- **URL**: https://github.com/llvm/circt/issues/8382
- **Match Reason**: String type on port causing crash, but different dialect (FIRRTL vs Moore) and different lowering pass

**Score Breakdown**:
| Category | Score |
|----------|-------|
| Dialect Match | 0 |
| Pass Match | 0 |
| Crash Type Match | 10 |
| Keyword Match | 25 |
| Assertion Pattern | 10 |

---

### #8176 - [MooreToCore] Crash when getting values to observe (Score: 35)
- **State**: OPEN
- **Labels**: Moore
- **URL**: https://github.com/llvm/circt/issues/8176
- **Match Reason**: Same dialect and pass, but different crash mechanism (unattached region vs type handling)

---

### #8211 - [MooreToCore] Unexpected observed values in llhd.wait (Score: 30)
- **State**: OPEN
- **Labels**: Moore
- **URL**: https://github.com/llvm/circt/issues/8211
- **Match Reason**: Same dialect and pass, but different issue (sensitivity list logic, not type conversion)

---

### #7627 - [MooreToCore] Unpacked array causes crash (Score: 30)
- **State**: CLOSED
- **Labels**: bug, Moore
- **URL**: https://github.com/llvm/circt/issues/7627
- **Match Reason**: Same dialect and pass, but different issue (unpacked array vs string type port)

---

### #8219 - [ESI] Assertion: dyn_cast on a non-existent value (Score: 25)
- **State**: CLOSED
- **Labels**: ESI
- **URL**: https://github.com/llvm/circt/issues/8219
- **Match Reason**: Same assertion message pattern (dyn_cast on non-existent value), but completely different dialect (ESI vs Moore)

## Analysis

### Why This Is NOT a Duplicate

The highest similarity score (55) is **below the duplicate threshold (70)**. 

While #8930 shares:
- ✅ Same dialect (Moore)
- ✅ Same pass (MooreToCore)
- ✅ Same assertion pattern (dyn_cast)

The root cause is **fundamentally different**:
- **#8930**: Crashes during `ConversionOp` lowering with `real` type conversion
- **Our bug**: Crashes during `getModulePortInfo()` with `string` type port → `sim::DynamicStringType`

### Unique Aspects of This Bug

1. **Crash Location**: `getModulePortInfo()` during port info construction
2. **Type Involved**: `sim::DynamicStringType` from sim dialect
3. **Failure Point**: `hw::InOutType` dyn_cast in `sanitizeInOut()`
4. **Trigger**: String type output port declaration (`output string str`)
5. **No Prior Reports**: No existing issue mentions `sanitizeInOut` or `PortImplementation.h`

## Recommendation

**✅ Proceed with new issue submission**

This is a genuinely new bug that has not been reported before. The crash path and root cause are distinct from all existing MooreToCore issues.
