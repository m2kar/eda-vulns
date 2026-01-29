---
name: circt-bug-reporter
description: "CIRCT project bug confirmation and GitHub issue generation with deep root cause analysis. Automates bug reproduction, root cause analysis (combining error logs, test code, and CIRCT source patterns), test case minimization, IEEE 1800-2005 compliance validation, duplicate issue detection, issue report creation, and optional submission to https://github.com/llvm/circt/issues. Use when processing crash logs from fuzzing tools (e.g., FeatureFuzz-SV output), confirming CIRCT toolchain bugs (circt-verilog, firtool, circt-opt, arcilator), or preparing bug reports for LLVM/CIRCT project. Input: crash directory path containing error.txt and source.sv."
---

# CIRCT Bug Reporter

Automated workflow for CIRCT bug confirmation, root cause analysis, and issue generation.

## Workflow Overview

```
Input: Crash directory (e.g., /edazz/FeatureFuzz-SV/output/crashes/assertion_xxx)
         ├── error.txt    (crash log with command and stack trace)
         └── source.sv    (test case that triggered the bug)

Step 1: Reproduce        → Confirm bug is reproducible with current toolchain
Step 2: Root Cause       → Deep analysis combining error logs, code, and CIRCT patterns
Step 3: Minimize         → Extract minimal test case using root cause insights
Step 4: Validate         → Verify test case correctness and classify bug type
Step 5: Check Duplicates → Search existing issues to avoid duplicates
Step 6: Generate         → Create issue.md with root cause analysis
Step 7: Submit           → (Optional, after human review) Create GitHub issue

Output: ./circt-b<id>/
        ├── bug.sv         (minimized test case)
        ├── error.log      (minimal error output)
        ├── command.txt    (single-line reproduction command)
        ├── analysis.json      (root cause analysis data)
        ├── root_cause.md      (root cause analysis report - AI generated)
        ├── minimize_report.md (minimization reasoning and verification - AI generated)
        ├── validation.json    (test case validation data)
        ├── validation.md      (validation report)
        ├── duplicates.json    (duplicate check results)
        ├── duplicates.md      (duplicate check report)
        └── issue.md           (generated issue report)
```

## Quick Start

```bash
# Set CIRCT toolchain path (if not in PATH)
export CIRCT_BIN=/opt/firtool/bin

# Step 1: Reproduce
python3 scripts/reproduce.py /path/to/crash/directory

# Step 2: Root Cause Analysis (AI-powered)
# Load skill: /root-cause-analysis
# Provide work directory: ./circt-b1
# Output: root_cause.md, analysis.json

# Step 3: Minimize Test Case (AI-powered)
# Load skill: /minimize-testcase
# Provide work directory: ./circt-b1
# Output: bug.sv, error.log, command.txt, minimize_report.md

# Steps 4-6: Automated scripts
python3 scripts/validate_testcase.py ./circt-b1
python3 scripts/check_duplicates.py ./circt-b1
python3 scripts/generate_issue.py ./circt-b1

# Review issue.md, then optionally submit
python3 scripts/submit_issue.py ./circt-b1
```

## Sub-Skill 1: Reproduce Bug

Verify the crash is reproducible with current CIRCT toolchain.

**Script**: `scripts/reproduce.py`

**Input**: Crash directory path
**Output**: Work directory `./circt-b<id>/` with:
- `source.sv` - copied test case
- `error.txt` - copied original error log
- `reproduce.log` - reproduction attempt output
- `metadata.json` - crash metadata and reproduction status

**Environment Variables**:
- `CIRCT_BIN` - Path to CIRCT binaries (default: search PATH for circt-verilog)

**Reproduction Logic**:
1. Parse `error.txt` to extract original command
2. Replace hardcoded paths with portable alternatives
3. Execute with current toolchain
4. Compare crash signatures (assertion message hash)

## Sub-Skill 2: Root Cause Analysis

Deep analysis combining error logs, test case code, and CIRCT source patterns using AI reasoning.

**Skill**: Load `/root-cause-analysis` skill to perform this step

**Input**: Work directory from Step 1
**Output**:
- `analysis.json` - structured analysis data
- `root_cause.md` - detailed root cause analysis report

**Invocation**:
```
Load skill: /root-cause-analysis
Provide work directory path (e.g., ./circt-b1)
```

**What the skill does** (AI-powered analysis):
1. **Parse Error Context**: Extract assertion, stack trace, crash location
2. **Analyze Test Case**: Identify constructs, patterns, problematic code
3. **Explore CIRCT Source**: Read relevant source files from `./circt-src`
4. **Correlate and Reason**: Form hypotheses connecting inputs to crash
5. **Generate Report**: Comprehensive `root_cause.md` with actionable insights

**Analysis Includes**:
- **Dialect Detection**: Identifies Moore, FIRRTL, Arc, etc.
- **Failing Pass**: Extracts which CIRCT pass crashed
- **Crash Pattern**: Matches against known crash categories
- **Test Case Analysis**: Identifies key constructs and features
- **CIRCT Source Analysis**: Reads and analyzes relevant compiler code
- **Hypotheses**: Generates ranked root cause hypotheses with evidence
- **Keywords**: Produces search terms for duplicate detection
- **Suggested Fix Directions**: Points to potential solutions

**CIRCT Source**: `./circt-src` (readonly)

**Crash Categories**:
| Category | Description |
|----------|-------------|
| Null/Invalid Value Access | dyn_cast on non-existent value |
| Legalization Failure | Failed to convert operation |
| SSA Violation | Value used outside definition scope |
| Type Mismatch | Type expectation not met |
| Incomplete Implementation | Unhandled case |

## Sub-Skill 3: Minimize Test Case (AI-Powered)

Create minimal reproducible test case using LLM reasoning and root cause insights.

**Skill**: Load `/minimize-testcase` skill (AI-powered) or use `scripts/minimize.py` (rule-based fallback)

**Input**: Work directory from Step 2 (must contain `analysis.json`)
**Output**:
- `bug.sv` (or `.fir`/`.mlir`) - minimized test case
- `error.log` - minimal error output  
- `command.txt` - single-line reproduction command
- `minimize_report.md` - minimization rationale and verification log

**AI-Powered Minimization Process**:

1. **Context Loading**:
   - Read `source.sv` (original test case)
   - Read `analysis.json` (root cause analysis, key features, hypotheses)
   - Read `error.txt` (original crash log with assertion message)
   - Read `metadata.json` (reproduction command, CIRCT version)

2. **Minimization Strategy Generation** (LLM Reasoning):
   - Analyze root cause hypotheses to identify **essential constructs**
   - Map key features to specific code lines/blocks
   - Identify **removable code**: unrelated signals, unused ports, comments
   - Generate conservative minimization plan preserving bug-triggering logic

3. **Iterative Minimization** (LLM-Guided):
   - Generate multiple candidate minimal test cases with decreasing complexity
   - For each candidate:
     - Write to temporary file
     - Execute reproduction command
     - Compare crash signature (assertion message hash)
   - Select smallest test case that reproduces identical crash

4. **Verification** (Strict):
   - ✅ **Crash reproduces** (exit code != 0)
   - ✅ **Same assertion message** (exact match or substring)
   - ✅ **Same crash location** (stack trace top frame)
   - ❌ If verification fails → retry with less aggressive minimization

5. **Output Generation**:
   - Write final `bug.sv`
   - Extract minimal error output → `error.log`
   - Simplify command → `command.txt`
   - Document minimization reasoning → `minimize_report.md`
   - Delete originals: `source.sv`, `error.txt`

**Minimization Principles**:
- **Preserve bug essence**: Keep all constructs mentioned in root cause hypotheses
- **Conservative approach**: When uncertain, keep code rather than remove
- **Verification-first**: Never output unverified test case
- **Readability**: Minimal code should still be clear and well-formatted

**Fallback to Rule-Based Script**:
If AI minimization fails or is unavailable, use `scripts/minimize.py`:
```bash
python3 scripts/minimize.py ./circt-b<id>
```
(Uses pattern-matching to preserve key constructs from `analysis.json`)

**Verification Requirements** (MANDATORY):
- ✅ Minimized test case MUST reproduce the crash
- ✅ Assertion message MUST match original (exact or substring)
- ✅ Exit code MUST be non-zero (crash/assertion failure)
- ❌ If any verification fails → do NOT output, retry with less minimization
- 📝 Document all verification attempts in `minimize_report.md`

**Example minimize_report.md Structure**:
```markdown
# Test Case Minimization Report

## Original Test Case
- Lines: 156
- Size: 4.2 KB
- Key features: packed union, dynamic array indexing, always_ff block

## Minimization Strategy
Based on root cause hypothesis: "Packed union lowering issue in MooreToCore"
- **Preserve**: union declaration, array indexing, always_ff sensitivity list
- **Remove**: unrelated signals (clk2, reset2), unused ports, comments

## Minimization Iterations
1. Candidate 1 (85 lines) - ✅ Verified (same assertion)
2. Candidate 2 (42 lines) - ✅ Verified (same assertion)
3. Candidate 3 (28 lines) - ❌ Failed (different crash location)
4. Final: Candidate 2 (42 lines) - ✅ Selected

## Verification Log
```bash
$ circt-verilog bug.sv --ir-moore | firtool --verilog
Assertion `isa<X>(Val) && "cast<Ty>() argument of incompatible type!"' failed.
```
Assertion message hash: a3b2c1d (MATCH ✅)
Stack trace top frame: MooreToCore.cpp:1234 (MATCH ✅)

## Final Result
- Reduction: 156 → 42 lines (73.1% reduction)
- Verification: PASSED ✅
- Command: `circt-verilog bug.sv --ir-moore | firtool --verilog`
```

## Sub-Skill 4: Validate Test Case

Verify test case correctness and classify the bug.

**Script**: `scripts/validate_testcase.py`

**Input**: Work directory from Step 3
**Output**:
- `validation.json` - structured validation data
- `validation.md` - validation report

**Validation Checks**:
1. **Syntax Issues**: Missing semicolons, unmatched begin/end
2. **Unsupported Features**: Dynamic arrays, classes, DPI
3. **Dialect Limitations**: Known Moore/FIRRTL limitations
4. **Cross-Tool Validation**: Tests with Verilator, Icarus, Slang (if available)

**Classification Results**:
| Result | Description |
|--------|-------------|
| `report` | Genuine bug, create issue |
| `report_as_feature_request` | Unsupported but valid SV feature |
| `check_existing_issue` | Matches known limitation with existing issue |
| `not_a_bug` | Design limitation or intentional |
| `fix_testcase` | Invalid test case |

**IEEE 1800-2005 Reference**: See `references/ieee_1800_2005_quick_ref.md`

## Sub-Skill 5: Check Duplicates

Search existing open issues using keywords from root cause analysis.

**Script**: `scripts/check_duplicates.py`

**Prerequisites**: `gh` CLI installed and authenticated

**Input**: Work directory with `analysis.json`
**Output**:
- `duplicates.json` - search results
- `duplicates.md` - duplicate check report

**Search Strategy**:
1. Generate queries from root cause analysis keywords
2. Search open issues with dialect label
3. Calculate similarity scores based on keyword matches
4. Rank by relevance

**Recommendations**:
| Recommendation | When |
|----------------|------|
| `review_existing` | High similarity issues found |
| `likely_new` | Related issues but low similarity |
| `new_issue` | No similar issues found |

## Sub-Skill 6: Generate Issue Report

Create structured issue report including root cause analysis.

**Script**: `scripts/generate_issue.py`

**Input**: Work directory from previous steps
**Output**: `issue.md` in work directory

**Issue Structure** (see `references/issue_template.md`):
```markdown
## Description
Brief description with root cause hypothesis

## Steps to Reproduce
1. Save test case as `test.sv`
2. Run: `<command>`

## Test Case
```sv
<bug.sv content>
```

## Error Output
```
<error.log content>
```

## Root Cause Analysis
- Dialect: <dialect>
- Failing Pass: <pass>
- Crash Category: <category>
- Hypotheses: <root cause hypotheses>

## Environment
- CIRCT Version: <version>

## Stack Trace
<collapsed stack trace>
```

## Sub-Skill 7: Submit Issue (Optional)

Submit issue to GitHub after human confirmation.

**Script**: `scripts/submit_issue.py`

**Prerequisites**:
- `gh` CLI authenticated
- Human review of `issue.md`
- Duplicate check passed

**Labels**:
- `bug` - always added
- Dialect label if identifiable (e.g., `Moore`, `FIRRTL`, `Arc`)

## CIRCT Tools Reference

| Tool | Purpose | Common Flags |
|------|---------|--------------|
| `circt-verilog` | SystemVerilog → MLIR | `--ir-hw`, `--ir-moore` |
| `firtool` | FIRRTL compiler | `--verilog`, `-O0` to `-O3` |
| `circt-opt` | MLIR optimizer | Pass pipelines |
| `arcilator` | Arc simulator | State lowering |

## Crash Type Identification

| Pattern | Likely Dialect/Tool |
|---------|---------------------|
| `MooreToCore` | Moore (circt-verilog) |
| `firrtl::` | FIRRTL (firtool) |
| `arc::` | Arc (arcilator) |
| `hw::` | HW dialect |
| `seq::` | Seq dialect |

## Work Directory Structure

```
./circt-b<id>/
├── bug.sv             # Minimized test case (final, AI-generated)
├── error.log          # Minimal error output (final)
├── command.txt        # Single-line reproduction command
├── analysis.json      # Root cause analysis data
├── root_cause.md      # Root cause analysis report (AI-generated)
├── minimize_report.md # Minimization reasoning and verification (AI-generated)
├── validation.json    # Test case validation data
├── validation.md      # Validation report
├── duplicates.json    # Duplicate check results
├── duplicates.md      # Duplicate check report
├── issue.md           # Generated issue report
├── metadata.json      # Workflow metadata
└── reproduce.log      # Reproduction output
```

## Decision Tree

```
                    ┌─ reproduce.py ──┐
                    │                 │
                    ▼                 ▼
              ✅ Reproduced      ❌ Not Reproduced
                    │                 │
                    ▼                 └──► STOP: Bug may be fixed
         /root-cause-analysis (AI skill)
                    │
                    ▼
         /minimize-testcase (AI skill)
                    │
                    ▼
         validate_testcase.py
                    │
          ┌────────┼────────┐
          ▼        ▼        ▼
       report  not_a_bug  fix_testcase
          │        │        │
          ▼        │        └──► Fix test and retry
   check_duplicates.py     │
          │                └──► STOP: Document as limitation
          ▼
    ┌─────┴─────┐
    ▼           ▼
review_existing new_issue
    │           │
    │           ▼
    │    generate_issue.py
    │           │
    │           ▼
    │    submit_issue.py
    │
    └──► Check existing issue first
```
