---
name: root-cause-analysis
description: "Deep root cause analysis for CIRCT compiler crashes using AI reasoning. Combines error logs, test case code, and CIRCT source code analysis to generate comprehensive root cause reports. Use when processing CIRCT crash directories that contain error.txt and source.sv. Input: work directory path (e.g., ./circt-b1). Output: root_cause.md with detailed analysis."
---

# Root Cause Analysis Skill

Deep root cause analysis for CIRCT compiler crashes using AI reasoning capabilities instead of fixed pattern matching.

## Overview

This skill performs comprehensive root cause analysis by:
1. Reading and understanding error logs (stack trace, assertion messages)
2. Analyzing test case code (SystemVerilog/FIRRTL)
3. Exploring relevant CIRCT source code
4. Reasoning about the relationship between inputs, crash patterns, and compiler internals
5. Generating detailed analysis reports with actionable insights

## Input Requirements

Work directory (from `reproduce.py`) containing:
- `error.txt` - Crash log with command and stack trace
- `source.sv` (or `.fir`/`.mlir`) - Test case that triggered the crash
- `metadata.json` - Reproduction metadata

## Output

- `root_cause.md` - Detailed root cause analysis report
- `analysis.json` - Structured analysis data for downstream tools

## Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    ROOT CAUSE ANALYSIS                          │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: Parse Error Context                                    │
│  ├── Extract assertion message                                  │
│  ├── Extract stack trace                                        │
│  ├── Identify failing pass/dialect                              │
│  └── Extract source file:line from crash                        │
├─────────────────────────────────────────────────────────────────┤
│  Step 2: Analyze Test Case                                      │
│  ├── Identify language (SV/FIRRTL/MLIR)                         │
│  ├── Identify key constructs used                               │
│  ├── Find potentially problematic patterns                      │
│  └── Understand test intent                                     │
├─────────────────────────────────────────────────────────────────┤
│  Step 3: Explore CIRCT Source Code                              │
│  ├── Locate crash site in source (./circt-src)                  │
│  ├── Read surrounding code context                              │
│  ├── Trace call path from stack frames                          │
│  └── Understand the failing logic                               │
├─────────────────────────────────────────────────────────────────┤
│  Step 4: Correlate and Reason                                   │
│  ├── Map test constructs to compiler handling                   │
│  ├── Identify gap between expected and actual behavior          │
│  ├── Form hypotheses about root cause                           │
│  └── Validate hypotheses against evidence                       │
├─────────────────────────────────────────────────────────────────┤
│  Step 5: Generate Report                                        │
│  ├── Executive summary                                          │
│  ├── Technical deep dive                                        │
│  ├── Ranked hypotheses with evidence                            │
│  └── Suggested fix directions                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Analysis Protocol

### Step 1: Parse Error Context

Read `error.txt` and extract:

```
EXTRACT FROM ERROR LOG:
1. Command line that triggered crash
2. Assertion message (if any): Assertion `...` failed
3. Error message: error: ..., fatal: ...
4. Stack trace frames (focus on circt::, mlir:: namespaces)
5. Source location: <file>.cpp:<line>
```

**Key patterns to identify:**
| Pattern | Meaning |
|---------|---------|
| `dyn_cast on a non-existent value` | Null pointer dereference in MLIR value |
| `failed to legalize operation` | Pass cannot convert operation |
| `use of value.*different block` | SSA dominance violation |
| `Assertion.*failed` | Internal invariant broken |
| `llvm_unreachable` | Unexpected code path reached |
| `unhandled case` | Missing switch/match handler |

### Step 2: Analyze Test Case

Read `source.sv` (or equivalent) and identify:

```
ANALYZE TEST CASE:
1. Module structure and hierarchy
2. Key language constructs:
   - Data types (unions, structs, enums, arrays)
   - Procedural blocks (always_ff, always_comb, initial)
   - Control flow (if/else, case, for/while loops)
   - Interfaces, packages, classes
   - Assertions, covergroups
3. Potentially problematic combinations:
   - Array indexing in sensitivity lists
   - Nested packed unions/structs
   - Complex parameterization
   - Unsupported SV features (DPI, classes, etc.)
4. Minimal reproducer characteristics
```

### Step 3: Explore CIRCT Source Code

**CIRCT source is available at: `./circt-src` (readonly)**

Navigate based on crash context:

```
CIRCT SOURCE EXPLORATION:
1. Locate failing pass directory:
   - Moore dialect: lib/Conversion/MooreToCore/, lib/Dialect/Moore/
   - FIRRTL dialect: lib/Dialect/FIRRTL/, lib/Conversion/FIRRTLToHW/
   - HW/SV/Seq/Comb: lib/Dialect/{HW,SV,Seq,Comb}/
   - Arc dialect: lib/Dialect/Arc/
   - LLHD dialect: lib/Dialect/LLHD/

2. From stack trace, locate exact crash file:line
   - Read function containing the crash
   - Understand the context and invariants expected
   
3. Trace the processing path:
   - How does the test construct get parsed?
   - How does it get lowered through passes?
   - Where does the handling break down?

4. Check for related patterns:
   - Similar operations that work
   - Edge cases in type handling
   - Missing visitor/handler implementations
```

**Key CIRCT directories:**
| Directory | Purpose |
|-----------|---------|
| `lib/Conversion/MooreToCore/` | SV to core dialect lowering |
| `lib/Dialect/Moore/` | Moore dialect definitions |
| `lib/Dialect/FIRRTL/Transforms/` | FIRRTL pass implementations |
| `lib/Conversion/FIRRTLToHW/` | FIRRTL to HW lowering |
| `lib/Conversion/ExportVerilog/` | HW to Verilog emission |
| `include/circt/Dialect/*/` | Dialect TableGen definitions |

### Step 4: Correlate and Reason

Apply reasoning to connect observations:

```
REASONING FRAMEWORK:
1. Input → Processing → Crash chain:
   - What specific input pattern triggers this?
   - Which pass/function processes this pattern?
   - Why does processing fail?

2. Expected vs Actual:
   - What should the compiler do with this input?
   - What is it actually doing?
   - Where does the divergence occur?

3. Hypothesis formation:
   - H1: [Most likely cause] - [Evidence]
   - H2: [Alternative cause] - [Evidence]
   - H3: [Less likely cause] - [Evidence]

4. Validation:
   - Does hypothesis explain all symptoms?
   - Are there contradicting observations?
   - What would confirm/refute hypothesis?
```

### Step 5: Generate Report

Output comprehensive `root_cause.md`:

```markdown
# Root Cause Analysis Report

## Executive Summary
[2-3 sentence summary of the bug and likely cause]

## Crash Context
- **Tool/Command**: [circt-verilog, firtool, etc.]
- **Dialect**: [Moore, FIRRTL, HW, etc.]
- **Failing Pass**: [pass name if identified]
- **Crash Type**: [Assertion, Segfault, etc.]

## Error Analysis

### Assertion/Error Message
```
[exact assertion or error message]
```

### Key Stack Frames
```
[relevant stack frames, filtered for circt/mlir]
```

## Test Case Analysis

### Code Summary
[Brief description of what the test case does]

### Key Constructs
- [construct 1]: [relevance to crash]
- [construct 2]: [relevance to crash]

### Potentially Problematic Patterns
[Specific patterns that may be causing the issue]

## CIRCT Source Analysis

### Crash Location
**File**: [filename.cpp]
**Function**: [function name]
**Line**: [approximate line]

### Code Context
```cpp
[relevant code snippet from CIRCT source]
```

### Processing Path
1. [First step in processing]
2. [Second step]
3. [Where it fails and why]

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: [Description]
**Evidence**:
- [Evidence point 1]
- [Evidence point 2]
**Mechanism**: [How this causes the crash]

### Hypothesis 2 (Medium Confidence)
[Similar structure]

## Suggested Fix Directions
1. [Suggestion 1 with rationale]
2. [Suggestion 2 with rationale]

## Keywords for Issue Search
`keyword1` `keyword2` `keyword3` ...

## Related Files to Investigate
- `path/to/file1.cpp` - [reason]
- `path/to/file2.cpp` - [reason]
```

## Tool Usage

### Required Tools
- `Read` - Read error.txt, source.sv, CIRCT source files
- `Grep` - Search patterns in CIRCT source
- `Glob` - Find relevant source files

### CIRCT Source Navigation
```bash
# Find pass implementations
grep -r "class.*Pass" ./circt-src/lib/

# Find dialect operation definitions
ls ./circt-src/include/circt/Dialect/Moore/

# Search for specific handling
grep -r "UnpackedUnionType" ./circt-src/lib/
```

## analysis.json Structure

```json
{
  "version": "2.0",
  "analysis_type": "ai_reasoning",
  "dialect": "Moore",
  "failing_pass": "MooreToCore",
  "crash_type": "assertion",
  "assertion_message": "...",
  "crash_location": {
    "file": "MooreToCore.cpp",
    "function": "...",
    "line": 123
  },
  "test_case": {
    "language": "systemverilog",
    "key_constructs": ["packed union", "array indexing"],
    "problematic_patterns": ["array in sensitivity list"]
  },
  "hypotheses": [
    {
      "description": "...",
      "confidence": "high",
      "evidence": ["...", "..."]
    }
  ],
  "keywords": ["keyword1", "keyword2"],
  "suggested_sources": [
    {"path": "...", "reason": "..."}
  ]
}
```

## Integration with circt-bug-reporter

This skill is called from the main `circt-bug-reporter` skill after the reproduce step:

```
reproduce.py → /root-cause-analysis → minimize.py → ...
```

The output `analysis.json` is consumed by:
- `minimize.py` - Uses keywords to preserve essential code
- `check_duplicates.py` - Uses keywords for issue search
- `generate_issue.py` - Includes analysis in issue report

## Example Invocation

When this skill is loaded, perform the analysis workflow:

```
Input: ./circt-b1/
├── error.txt      (crash log)
├── source.sv      (test case)
└── metadata.json  (from reproduce.py)

Actions:
1. Read error.txt → extract crash context
2. Read source.sv → analyze test case
3. Based on stack trace, read relevant CIRCT source files
4. Reason about the relationship
5. Generate root_cause.md and analysis.json

Output: ./circt-b1/
├── root_cause.md   (detailed analysis report)
├── analysis.json   (structured data)
└── metadata.json   (updated with analysis summary)
```

## Quality Checklist

Before completing analysis:
- [ ] Error context fully extracted (assertion, stack trace)
- [ ] Test case constructs identified
- [ ] At least one CIRCT source file examined
- [ ] Hypotheses have supporting evidence
- [ ] Keywords are actionable for issue search
- [ ] Report is actionable for developers
