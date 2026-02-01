# Duplicate Issue Check Report

**Crash ID**: 260128-00000b82  
**Report Generated**: 2026-01-31T22:55:00Z  
**Status**: Phase 2 (Check Duplicates) In Progress

---

## 1. Search Strategy

### Keywords Extracted from analysis.json
- **Key Functions**: `sanitizeInOut`, `getModulePortInfo`, `SVModuleOpConversion`
- **Types**: `string array`, `unpacked array`, `InOutType`, `UnpackedArrayType`
- **Error Signature**: `dyn_cast on a non-existent value`
- **Location**: `PortImplementation.h:177`
- **Dialect**: `moore` (MooreToCore conversion)

### Search Queries Executed
1. `sanitizeInOut string array` 
2. `InOutType assertion`
3. `MooreToCore crash`
4. `unpacked array`
5. `dyn_cast`

---

## 2. Duplicate Check Results

### Top 5 Most Similar Issues

#### 1. **#8219** - CLOSED ⚠️
**Title**: `[ESI]Assersion: dyn_cast on a non-existent value`  
**State**: CLOSED (2025-02-26)  
**Similarity Score**: **8.5/10**

**Matched Keywords**:
- ✅ Exact error message: "dyn_cast on a non-existent value"
- ✅ Same assertion failure pattern
- ❌ Different subsystem (ESI, not MooreToCore)

**Analysis**:
This issue has the EXACT same error message as our crash. The root cause is unsafe `dyn_cast` on potentially null/non-existent values. Although it's in the ESI subsystem rather than MooreToCore, the pattern is identical and suggests a systemic issue with how CIRCT handles null/non-existent values in type casts.

**Status**: CLOSED - Not a candidate for duplicate filing

---

#### 2. **#7627** - CLOSED ⚠️
**Title**: `[MooreToCore] Unpacked array causes crash`  
**State**: CLOSED (2024-09-25)  
**Similarity Score**: **8.0/10**

**Matched Keywords**:
- ✅ MooreToCore subsystem
- ✅ Unpacked array handling
- ✅ Crash on conversion
- ✅ Same dialect context

**Analysis**:
This is a MooreToCore crash specifically with unpacked arrays. Our crash involves unpacked array of strings, which is directly in this issue's scope. The issue was CLOSED, possibly as WONTFIX or duplicate of a broader feature gap.

**Status**: CLOSED - Related but resolved/tracked elsewhere

---

#### 3. **#4036** - OPEN 🔴
**Title**: `[PrepareForEmission] Crash when inout operations are passed to instance ports`  
**State**: OPEN (2022-09-30)  
**Similarity Score**: **7.5/10**

**Matched Keywords**:
- ✅ Assertion failure with InOutType
- ✅ Port/instance processing
- ✅ Same type handling issue
- ❌ Different conversion phase (PrepareForEmission vs MooreToCore)

**Analysis**:
This OPEN issue involves InOutType assertion failures during port processing. While it occurs in a different phase (PrepareForEmission rather than MooreToCore), the underlying issue is similar: improper handling of port types leading to type mismatches. The issue has been open since 2022, suggesting this is a known problem area.

**Status**: OPEN - May be related but in different phase

---

#### 4. **#8276** - OPEN 🔴
**Title**: `[MooreToCore] Support for UnpackedArrayType emission`  
**State**: OPEN (2025-03-04)  
**Similarity Score**: **7.0/10** (Most Relevant OPEN Issue)

**Matched Keywords**:
- ✅ MooreToCore subsystem
- ✅ UnpackedArrayType support
- ✅ Feature gap causing crashes
- ✅ Directly addresses the root cause

**Analysis**:
This is the MOST RELEVANT OPEN issue. It's a feature request to support `UnpackedArrayType` emission in the MooreToCore pass. The issue explicitly states that currently, unpacked arrays are converted to packed arrays, losing semantics. When attempting to preserve unpacked arrays, it raises "failed to legalize operation" errors.

**Our crash** involves an unpacked array of strings as a module port. The Moore-to-Core conversion yields a null type (empty mlir::Type) which then causes the `sanitizeInOut()` dyn_cast to fail. This is exactly the type of issue described in #8276.

**Status**: OPEN - **Primary candidate for duplicate/related issue**

---

#### 5. **#8215** - OPEN 🔴
**Title**: `[MooreToCore] OOB array slices of unpacked arrays lowered like packed arrays`  
**State**: OPEN (2025-02-12)  
**Similarity Score**: **6.5/10**

**Matched Keywords**:
- ✅ MooreToCore subsystem
- ✅ Unpacked array handling
- ✅ Array conversion issues

**Analysis**:
Related to unpacked array handling in MooreToCore. Less directly relevant than #8276 but part of the same feature gap.

**Status**: OPEN - Related to same feature area

---

## 3. Recommendation

### **Verdict: `review_existing` ⚠️**

**Confidence Level**: HIGH

### Rationale

1. **Exact Error Message Match**: Issue #8219 has the EXACT same error message ("dyn_cast on a non-existent value"), but it's CLOSED and in a different subsystem (ESI).

2. **Primary Root Cause**: Issue #8276 is the most relevant OPEN issue. It directly addresses the missing `UnpackedArrayType` support in MooreToCore, which is the underlying root cause of our crash.

3. **Specific to Ports**: Our crash is specifically in port type conversion (`sanitizeInOut()` in `PortImplementation.h:177`). While #8276 addresses the broader UnpackedArrayType support, the port-specific aspect may warrant a new issue OR tracking under #8276.

4. **Closed Related Issues**: Issues #7627 and partial patterns from #8219 suggest this area has been investigated before, but the fundamental issue (UnpackedArrayType support) remains open.

### Recommended Action

✅ **File a NEW issue**, but reference #8276 as related/blocking feature.

**Rationale**: 
- The current crash is a specific manifestation of the UnpackedArrayType support gap (#8276)
- While #8276 is marked as a feature request, our crash is a critical bug that should be explicitly filed
- The port-specific nature (PortImplementation.h) and string array specificity ("output string s[1:0]") is more specific than the general UnpackedArrayType issue
- Linking to #8276 as a blocker helps CIRCT developers understand the dependency

### Similarity Score Summary
- **Top Score**: 8.5 (but CLOSED and different subsystem)
- **Highest OPEN Match**: 7.0 (#8276 - UnpackedArrayType feature)
- **Threshold for Duplicate**: 10.0+ (none meet this)

---

## 4. Metadata

- **Total Issues Found**: 5 relevant results
- **OPEN Issues**: 3
- **CLOSED Issues**: 2
- **Most Recent**: #8276 (2025-03-04)
- **Oldest Related**: #4036 (2022-09-30)

---

## 5. Next Steps

1. ✅ Create new GitHub issue report
2. ✅ Reference #8276 as related/blocking feature
3. ✅ Include this analysis in issue body
4. ✅ Tag with `Moore`, `MooreToCore`, `bug` labels

