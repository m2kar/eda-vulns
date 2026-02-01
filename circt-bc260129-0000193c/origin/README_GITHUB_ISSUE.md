# CIRCT GitHub Issue Report - Final Deliverable

**Status**: ✅ **COMPLETE AND READY FOR SUBMISSION**

---

## 📄 Main Output File

### `issue.md` - Complete GitHub Issue Report
- **Size**: 7.4 KB
- **Lines**: 173
- **Format**: GitHub-compatible Markdown
- **Status**: ✅ Ready for submission to https://github.com/llvm/circt/issues/new

---

## 📋 Report Structure

The generated GitHub issue includes all required sections for a comprehensive CIRCT bug report:

### 1. **Title** (Line 1)
```
[LLHD] Crash when using `real` type in sequential logic blocks
```

### 2. **Description** (Lines 3-7)
- Problem statement
- Context for related issues (#8930, #8269)
- Indication that this is a specific crash scenario

### 3. **Steps to Reproduce** (Lines 9-26)
- Minimal test case (4 lines of code)
- Exact reproduction command
- How to observe the crash

### 4. **Expected Behavior** (Lines 28-34)
- What should happen (proper compilation or clear error)
- What the user expects instead of a crash

### 5. **Actual Behavior** (Lines 36-54)
- Current error manifestation
- Original error (assertion failure)
- Explanation of the bitwidth issue

### 6. **Root Cause Analysis** (Lines 56-96)
- **Technical Details** (4-point breakdown):
  1. Type conversion failure
  2. Bitwidth calculation issue
  3. Invalid IntegerType creation
  4. Missing type support
- **Code snippets** showing the problem
- **Call stack trace** from the crash

### 7. **Environment** (Lines 98-106)
- CIRCT version: firtool-1.139.0
- Toolchain: llvm-22
- Affected passes and files

### 8. **Validation** (Lines 108-115)
- ✅ Syntactically valid (Verilator, Slang)
- ✅ Reproducible on CIRCT
- ✅ Valid bug (not user error)

### 9. **Related Issues** (Lines 117-131)
- **#8930**: [MooreToCore] Crash with sqrt/floor
  - Similarity: 8/10
  - Same root cause: `hw::getBitWidth()` failing for Float64Type
  
- **#8269**: [MooreToCore] Support `real` constants
  - Similarity: 8/10
  - Incomplete `real` type support
  
- **Note**: 8/10 is below the 10.0 threshold for exact duplicate
- **Distinction**: This issue is about `real` in sequential logic (always_ff)

### 10. **Suggested Fixes** (Lines 133-151)
- **Short-term**: Type validation to reject non-promotable types
  - Add check in `Promoter::insertBlockArgs()`
  - Emit clear error message
  
- **Long-term**: Proper support options
  - Extend `hw::getBitWidth()` for floating-point types
  - Add MooreToCore conversion rules
  - Implement proper LLHD support or document limitations

### 11. **Additional Context** (Lines 153-173)
- Crash ID: 260129-0000193c
- Severity: High
- Reproducibility: Deterministic
- Impact assessment
- Critical points summary

---

## 🔍 Technical Highlights

### The Bug
CIRCT crashes when processing SystemVerilog `real` (floating-point) types in sequential logic:

```systemverilog
module test(input logic clk);
  real r;
  always_ff @(posedge clk) r <= 1.0;
endmodule
```

### Root Cause
1. `real` → parsed as Moore dialect → converted to Float64Type
2. `hw::getBitWidth()` returns -1 for Float64Type (unknown)
3. -1 interpreted as unsigned becomes 0xFFFFFFFF
4. After masking: 1073741823 (2^30 - 1)
5. Exceeds max hardware bitwidth of 16777215 (2^24 - 1)
6. Assertion failure or validation error

### Why It Matters
- Users get confusing error messages with impossible bitwidths
- Valid SystemVerilog syntax causes compiler crash
- Blocks testbenches and behavioral models using `real` types
- Related to multiple outstanding issues (#8930, #8269)

---

## ✅ Quality Assurance

### Validation Results
- ✅ **Syntax Validity**: PASS (Verilator, Slang both accept it)
- ✅ **Bug Reproducibility**: PASS (consistently crashes on CIRCT)
- ✅ **Cross-Tool Validation**: PASS (other synthesis tools accept it)
- ✅ **Classification**: REPORT (valid bug, not invalid testcase)

### Duplicate Analysis
- **Related Issues**: #8930 (8/10), #8269 (8/10)
- **Threshold for Duplicate**: 10.0
- **Status**: NOT EXACT DUPLICATE (but related)
- **Justification**: This issue demonstrates a specific manifestation in sequential logic blocks that merits separate tracking

### Report Completeness
- ✅ Clear title
- ✅ Problem description
- ✅ Minimal reproducible example
- ✅ Exact reproduction command
- ✅ Expected vs actual behavior
- ✅ Technical root cause analysis
- ✅ Code evidence and stack trace
- ✅ Environment information
- ✅ Validation confirmation
- ✅ Related issues analysis
- ✅ Suggested fixes (short and long term)
- ✅ Additional context

---

## 🚀 How to Submit

### Step 1: Copy the Report
```bash
cat /home/zhiqing/edazz/eda-vulns/circt-bc260129-0000193c/origin/issue.md
```

### Step 2: Visit GitHub
Go to: https://github.com/llvm/circt/issues/new

### Step 3: Create Issue
1. Click "New Issue"
2. Paste the entire contents of `issue.md` into the body
3. Title is already provided: `[LLHD] Crash when using 'real' type in sequential logic blocks`

### Step 4: Add Labels (Optional but Recommended)
- `bug` - This is a bug
- `crash` - Causes compiler crash
- `LLHD` - LLHD dialect issue
- `MooreToCore` - Type conversion related
- `type-support` - Type system issue

### Step 5: Submit
Click "Submit new issue"

---

## 📚 Supporting Documentation

Also included in the same directory:

1. **GENERATION_REPORT.md** (6.8 KB)
   - Comprehensive report on the issue generation process
   - Quality metrics
   - Validation details

2. **ISSUE_SUMMARY.txt** (4.0 KB)
   - Quick reference guide
   - Key information summary
   - Status overview

3. **root_cause.md** (4.9 KB)
   - Original root cause analysis
   - Detailed technical breakdown
   - Component relationships

4. **validation.md** (3.4 KB)
   - Validation test results
   - Cross-tool comparison
   - Feature analysis

5. **duplicates.md** (8.0 KB)
   - Duplicate issue search results
   - Similarity scoring analysis
   - Related issues comparison

6. **bug.sv** (290 bytes)
   - Minimal test case
   - 4 lines of code

---

## 💡 Key Differentiators

This issue stands out because:

1. **Minimal Test Case**: Just 4 lines of code
2. **Clear Reproduction**: Single command to reproduce
3. **Technical Depth**: Explains the exact overflow mechanism
4. **Validation**: Confirmed as real bug through cross-tool validation
5. **Relatedness**: Properly analyzes connection to #8930 and #8269
6. **Actionability**: Provides both quick and proper fixes
7. **Evidence**: Includes code snippets and stack traces
8. **Impact**: Explains user-facing consequences

---

## 📊 Comparison with Related Issues

| Aspect | #8930 | #8269 | This Issue |
|--------|-------|-------|-----------|
| **Context** | Real in function calls | Real constants | Real in sequential logic |
| **Phase** | MooreToCore | MooreToCore | Mem2Reg |
| **Manifestation** | Crash with sqrt/floor | Unsupported feature | Invalid bitwidth error |
| **Root Cause** | Float64Type handling | Incomplete support | Mem2Reg type validation |
| **Similarity** | 8/10 | 8/10 | - |
| **Exact Duplicate** | No | No | - |

**Conclusion**: All three issues stem from incomplete `real` type support in CIRCT, but each demonstrates a different manifestation that merits tracking.

---

## 🎯 Next Steps for CIRCT Maintainers

1. **Investigate** whether recent fixes to #8930 or #8269 might address this issue
2. **Verify** if this still occurs on the latest main branch
3. **Triage** as related to the broader `real` type support initiative
4. **Fix** by implementing one of the suggested solutions:
   - Short-term: Add type validation in Mem2Reg
   - Long-term: Proper Float64Type support throughout the pipeline

---

## 📝 File Information

```
File: issue.md
Location: /home/zhiqing/edazz/eda-vulns/circt-bc260129-0000193c/origin/
Size: 7.4 KB
Lines: 173
Format: GitHub Markdown
Status: ✅ Ready for submission
```

---

**Generated**: 2026-02-01  
**Crash ID**: 260129-0000193c  
**Status**: ✅ Complete and Ready for GitHub Submission
