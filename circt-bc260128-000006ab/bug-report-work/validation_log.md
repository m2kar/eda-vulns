# Validation Log

## Test Case Validation

### Original Test Case (test.sv)
**Status:** ✅ Compiles successfully on current toolchain
**Toolchain:** CIRCT firtool-1.139.0, LLVM 22.0.0git
**Expected:** Crash (assertion failure)
**Actual:** No crash
**Conclusion:** Bug appears to be fixed in current build

### Minimal Test Cases

| Test Case | Lines | Expected | Actual | Status |
|-----------|-------|----------|--------|--------|
| minimal_1.sv | 6 | Crash | No crash | ✅ Fixed |
| minimal_2.sv | 5 | Crash | No crash | ✅ Fixed |
| minimal_3.sv | 6 | Crash | No crash | ✅ Fixed |

## Validation Summary

**Test Date:** Sat Jan 31 19:09:59 UTC 2026

**Environment:**
- Tool: arcilator (CIRCT firtool-1.139.0)
- Backend: LLVM 22.0.0git
- Platform: Linux x86_64

**Results:**
- Original crash NOT reproducible on current toolchain
- All minimal variants compile without errors
- Test case produces valid LLVM IR output
- No assertion failures or crashes observed

**Implications:**
1. The bug (StateType with LLHD reference type) has been fixed
2. Fix is present in CIRCT firtool-1.139.0
3. inout ports with tri-state assignments now work correctly
4. The original crash report documents a now-fixed issue

**Recommendation:**
- Submit the issue as a bug report documenting the crash
- Include minimal test case (minimal_1.sv)
- Note that bug is fixed in current version
- Suggest testing on older toolchain to verify fix timeline
