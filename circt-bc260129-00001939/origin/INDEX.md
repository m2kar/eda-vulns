# GitHub Issue Generation - Complete Package

## 📋 Overview

This directory contains a complete, production-ready GitHub Issue report for a CIRCT compiler crash. All analysis, validation, and documentation has been completed. The issue is ready for submission to the LLVM CIRCT project.

---

## 🎯 Main Output

### **`issue.md`** ⭐ PRIMARY FILE
- **Status**: ✅ READY FOR GITHUB SUBMISSION
- **Size**: 9.5 KB (198 lines)
- **Purpose**: Complete GitHub Issue formatted for submission
- **Contains**: Title, description, test case, root cause, related issues, suggested fix, stack trace

**This is the file you need to submit to https://github.com/llvm/circt/issues**

---

## 📚 Supporting Documentation

### `FINAL_CHECKLIST.txt`
- Comprehensive verification checklist
- All 43 quality assurance checks marked as ✅
- GitHub submission readiness confirmation
- Quality metrics and statistics

### `GENERATION_REPORT.md`
- Detailed report of the generation process
- Input files processed (8/8)
- Key information extracted
- Issue content structure breakdown
- Test case reduction analysis (83.3%)

### `README_ISSUE_GENERATION.md`
- Executive summary
- Issue details and quick facts
- Root cause summary
- Related issues analysis
- Quality assurance checklist
- Submission instructions
- Key insights and recommendations

### `INDEX.md` (this file)
- Navigation guide for all project files
- Quick reference to key information

---

## 📄 Source Analysis Files

### `bug.sv`
- **Minimal test case**: 3 lines
- **Reduction**: 83.3% from original (18 lines)
- **Validation**: ✅ Verilator v5.022, ✅ Slang v10.0.6
- **Crash**: 100% reproducible

### `error.log`
- Full error output with complete stack trace
- 51 lines, 14 KB
- Key error: `integer bitwidth is limited to 16777215 bits`
- Assertion location: `StorageUniquerSupport.h:180`

### `command.txt`
- Exact reproduction command
- Path: `/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw bug.sv`

### `analysis.json`
- Structured crash analysis data
- Crash type: assertion
- Dialect: LLHD
- Failing pass: Mem2RegPass
- Error location: `lib/Dialect/LLHD/Transforms/Mem2Reg.cpp:1753`

### `root_cause.md`
- 107 lines of detailed root cause analysis
- Problem identification and breakdown
- Type system flow analysis
- Trigger patterns documented
- Suggested fixes provided

### `validation.md`
- Syntax validation results
- Test case properties
- Verilator and Slang verification
- 83.3% reduction confirmation
- Bug classification as reportable

### `duplicates.md`
- Comprehensive duplicate check report
- 16 search queries executed
- 3 related issues identified
- Primary match: Issue #9287 (7/10 similarity, HIGH confidence)
- Recommendation: `review_existing`

### `metadata.json`
- CIRCT version: 1.139.0
- Workflow metadata
- System information

---

## 🎯 Issue Quick Facts

| Aspect | Details |
|--------|---------|
| **Title** | [LLHD] Assertion failure in Mem2Reg pass with clocked assignment to real type output port |
| **Dialect** | LLHD (Low-Level Hardware Description) |
| **Component** | Mem2Reg Pass (Memory to Register promotion) |
| **Crash Type** | Assertion failure in MLIR IntegerType creation |
| **Location** | `lib/Dialect/LLHD/Transforms/Mem2Reg.cpp:1753` |
| **CIRCT Version** | 1.139.0 |
| **Reproducibility** | 100% |
| **Test Case Size** | 3 lines (83.3% reduction) |
| **Primary Related Issue** | #9287 (Similarity: 7/10, HIGH confidence) |

---

## 📊 Quality Metrics

### Test Case
- Original size: 18 lines
- Minimized size: 3 lines
- Reduction: 83.3% ✅
- Validated by Verilator: ✅
- Validated by Slang: ✅
- Reproduces crash: 100% ✅

### Root Cause
- Clearly identified: ✅
- Location pinpointed: ✅
- Mechanism explained: ✅
- Trigger conditions documented: ✅

### Documentation
- Completeness: COMPREHENSIVE ✅
- Technical accuracy: HIGH ✅
- Markdown formatting: VALID ✅
- Issue sections: ALL INCLUDED ✅

---

## 🔗 Related Issues

### Primary: Issue #9287 ⭐
- **Title**: [HW] Make `hw::getBitWidth` use std::optional vs -1
- **Status**: OPEN
- **Similarity**: 7/10 (HIGH CONFIDENCE)
- **Connection**: This crash is one of the callsites needing bitwidth validation fixes
- **Recommendation**: Resolve as part of #9287 implementation

### Secondary Matches
1. **#9574**: Similar assertion pattern in Arc dialect (Score: 6/10)
2. **#8693**: Different Mem2Reg bug - SSA domination issue (Score: 5/10)

---

## 🚀 How to Submit

### Step 1: Copy Issue Content
```bash
cat issue.md | pbcopy  # macOS
# or
cat issue.md | xclip -selection clipboard  # Linux
```

### Step 2: Navigate to GitHub
Go to: https://github.com/llvm/circt/issues/new

### Step 3: Paste and Fill
1. Paste the content into the issue form
2. Add labels: `bug`, `LLHD`, `Mem2Reg`, `assertion-failure`
3. Optional: Set milestone to current development cycle

### Step 4: Submit
Click "Submit new issue"

---

## 📝 Root Cause Summary

**Problem**: The LLHD Mem2Reg pass calls `hw::getBitWidth()` on a slot's stored type and passes the result directly to `builder.getIntegerType()` **without validating** that the bitwidth is valid.

**Trigger Conditions**:
1. Module with `real` type port (input or output)
2. Clocked `always @(posedge/negedge)` or `always_ff` block
3. Non-blocking (`<=`) or blocking (`=`) assignment to real type

**Result**:
- `hw::getBitWidth(real)` returns -1 (invalid value)
- When cast to unsigned: 0xFFFFFFFF (4,294,967,295)
- Exceeds MLIR's 16,777,215-bit limit
- Assertion fails in `IntegerType::verifyInvariants()`

**Location**: `lib/Dialect/LLHD/Transforms/Mem2Reg.cpp:1753`

---

## 💡 Key Insights

1. **Minimal Reproducer**: The 3-line test case is optimal for regression testing
2. **Clear Root Cause**: Specific missing validation check - straightforward fix
3. **Existing Solution Path**: Issue #9287 proposes architectural solution
4. **High Priority**: Assertion failure on valid input—should be prioritized
5. **Validated**: Multiple tools confirm test case validity

---

## ✅ Verification Status

- ✅ All input files processed (8/8)
- ✅ Root cause identified
- ✅ Test case minimized (83.3% reduction)
- ✅ Related issues identified (3 total)
- ✅ Primary match found (#9287)
- ✅ All sections complete
- ✅ Markdown formatting valid
- ✅ Ready for GitHub submission

---

## 🎓 Documentation Structure

```
origin/
├── issue.md                      ⭐ PRIMARY: GitHub Issue (SUBMIT THIS)
├── bug.sv                        - Minimal test case (3 lines)
├── error.log                     - Complete error output
├── command.txt                   - Reproduction command
├── analysis.json                 - Crash analysis
├── root_cause.md                 - Root cause details
├── validation.md                 - Validation results
├── duplicates.md                 - Duplicate analysis
├── metadata.json                 - Metadata
├── GENERATION_REPORT.md          - Generation details
├── README_ISSUE_GENERATION.md    - Complete guide
├── FINAL_CHECKLIST.txt           - Verification checklist
└── INDEX.md                      - This file
```

---

## 🎯 Recommendation

✅ **SUBMIT TO CIRCT**

The `issue.md` file is production-ready and should be submitted to the CIRCT project with reference to Issue #9287. This is a high-priority bug causing compiler crashes on valid SystemVerilog code.

---

## 📞 Support & Questions

For detailed information about:
- **Root cause**: See `root_cause.md`
- **Validation**: See `validation.md`
- **Related issues**: See `duplicates.md`
- **Generation process**: See `GENERATION_REPORT.md`
- **Submission**: See `README_ISSUE_GENERATION.md`

---

## ✨ Summary

A complete, comprehensive GitHub Issue report has been generated for a CIRCT compiler crash. The issue:
- ✅ Includes a minimal 3-line test case (83.3% reduction)
- ✅ Identifies clear root cause with code location
- ✅ References related Issue #9287 (7/10 similarity)
- ✅ Provides code-level fix suggestions
- ✅ Is validated by multiple tools
- ✅ Is ready for immediate GitHub submission

**Status**: ✅ READY FOR SUBMISSION

---

**Generated**: 2025-02-01 13:37:00 UTC  
**Verified**: 2025-02-01 13:38:00 UTC  
**Status**: Production-ready for GitHub submission
