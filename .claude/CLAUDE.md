# EDA-Vulns Project Memory

## Project Overview

Repository for vulnerability research and CVE submissions for EDA (Electronic Design Automation) tools, focusing on hardware compilers, synthesis tools, and simulation platforms.

**Target tools**: CIRCT, Yosys, Verilator, and other open-source EDA tools.

## CIRCT Bug Reporter Skill

### Design Philosophy

The `circt-bug-reporter` skill automates the complete workflow from fuzzer-generated crash to GitHub issue submission. It follows the principle of **progressive automation with human checkpoints**.

### Workflow Architecture

```
Fuzzer Output → Reproduce → Minimize → Generate Issue → [Human Review] → Submit
     (Input)       ↓           ↓            ↓              ↓            ↓
                circt-b<id>/ work directory (stateful workflow)
```

**Key Design Decisions:**

1. **Stateful Work Directory**: Each bug gets a unique `circt-b<id>/` directory containing:
   - Original files (error.txt, source.sv)
   - Minimized artifacts (minimal.sv, command.txt)
   - Metadata (metadata.json)
   - Generated report (issue.md)
   - Logs (reproduce.log)

2. **Pipeline Modularity**: Each step is an independent Python script that can be run separately or chained:
   - `reproduce.py` - Verifies bug reproducibility
   - `minimize.py` - Creates minimal test case
   - `generate_issue.py` - Formats issue report
   - `submit_issue.py` - Submits to GitHub

3. **Human-in-the-Loop**: Mandatory review before submission at Step 4

### Input Format

Expected directory structure from fuzzing tools (e.g., FeatureFuzz-SV):

```
/path/to/crash/directory/
├── error.txt          # Crash log with:
│                      # - Crash Type: assertion/timeout/segfault
│                      # - Hash: <crash_signature>
│                      # - Original Program File: <name>
│                      # - Compilation Command: <full_command>
│                      # - Error Message: <stderr_output>
│                      # - Stack dump: <backtrace>
└── source.sv          # SystemVerilog test case that triggered bug
```

### Sub-Skills Breakdown

#### Sub-Skill 1: Reproduce (`reproduce.py`)

**Purpose**: Verify crash reproducibility with current toolchain

**Logic**:
1. Parse `error.txt` to extract:
   - Crashing tool (circt-verilog/firtool/arcilator)
   - Original command pipeline
   - Assertion message
2. Adapt command for current environment:
   - Replace hardcoded paths with current CIRCT_BIN
   - Truncate pipeline after crashing tool
   - Convert to portable paths
3. Execute and compare crash signatures:
   - Hash based on: assertion message + key stack frames (skip first 10)
   - Fuzzy matching allows version differences
4. Output: Work directory with metadata.json

**Environment Variables**:
- `CIRCT_BIN`: Path to CIRCT binaries (default: search PATH)

#### Sub-Skill 2: Minimize (`minimize.py`)

**Purpose**: Extract minimal reproducible test case

**Strategy**:
1. **Code Minimization**:
   - Line-by-line deletion (preserve crash)
   - Remove comments and whitespace
   - Simplify types where possible
2. **Command Simplification**:
   - Remove post-crash pipeline stages
   - Strip output file options (-o)
   - Use relative paths (test.sv)
3. Verification: Re-run after each change to ensure crash persists

**Output**: `minimal.sv`, `command.txt`

#### Sub-Skill 3: Generate Issue (`generate_issue.py`)

**Purpose**: Create formatted GitHub issue report

**Format** (follows CIRCT conventions):
```markdown
# [Dialect/Tool] Brief description

## Description
One-line summary

## Steps to Reproduce
1. Save code as `test.sv`
2. Run: `<command>`

## Test Case
```systemverilog
<minimal code>
```

## Error Output
<key error message>

## Environment
- CIRCT Version: <version>

<details>
<summary>Stack Trace</summary>
<full trace>
</details>
```

**Dialect Detection**: Auto-detect from stack trace patterns:
- `MooreToCore` → Moore dialect
- `firrtl::` → FIRRTL dialect
- `arc::` → Arc dialect
- etc.

**Labels**: Automatically assign `bug` + dialect label

#### Sub-Skill 4: Submit (`submit_issue.py`)

**Purpose**: Submit to GitHub after human confirmation

**Prerequisites**:
- `gh` CLI authenticated (`gh auth login`)
- `issue.md` reviewed

**Modes**:
- `--dry-run`: Preview without submitting
- Interactive: Requires `y` confirmation

**Output**: GitHub issue URL stored in metadata.json

### CIRCT Tools Reference

| Tool | Purpose | Crash Patterns |
|------|---------|----------------|
| `circt-verilog` | SystemVerilog → MLIR | Moore dialect issues |
| `firtool` | FIRRTL compiler | FIRRTL dialect issues |
| `circt-opt` | MLIR optimizer | Pass infrastructure |
| `arcilator` | Arc simulator | Arc dialect issues |

### Metadata Schema

`metadata.json` structure:

```json
{
  "crash_type": "assertion",
  "hash": "03ce98b35955",
  "original_file": "program_xxx.sv",
  "original_command": "...",
  "assertion_message": "...",
  "crashing_tool": "circt-verilog",
  "stack_trace": "...",
  "reproduction": {
    "command": "...",
    "circt_bin": "/opt/firtool/bin",
    "circt_version": "firtool-1.139.0",
    "timestamp": "2026-01-27T14:20:00",
    "reproduced": true,
    "same_crash": true,
    "original_signature": "4971a8320365",
    "repro_signature": "4e7c6ea6d4b7"
  },
  "minimization": {
    "original_lines": 50,
    "minimal_lines": 14,
    "simplified_command": "circt-verilog --ir-hw test.sv"
  },
  "issue": {
    "title": "[Moore] Assertion failed...",
    "labels": ["bug", "Moore"],
    "dialect": "Moore"
  },
  "submitted": {
    "url": "https://github.com/llvm/circt/issues/9999",
    "title": "...",
    "labels": [...]
  }
}
```

### Usage Examples

**Basic workflow**:
```bash
export CIRCT_BIN=/opt/firtool/bin

# Reproduce
python3 scripts/reproduce.py /edazz/FeatureFuzz-SV/output/crashes/assertion_xxx

# Minimize
python3 scripts/minimize.py ./circt-b1

# Generate issue
python3 scripts/generate_issue.py ./circt-b1

# Review issue.md manually

# Submit (optional)
python3 scripts/submit_issue.py ./circt-b1
```

**Dry-run submission**:
```bash
python3 scripts/submit_issue.py ./circt-b1 --dry-run
```

### Testing Notes

Tested with real crash from FeatureFuzz-SV:
- Crash: `assertion_03ce98b35955_20260125_204541`
- Bug: dyn_cast assertion on packed union module ports
- Dialect: Moore (SystemVerilog)
- Result: Successfully generated valid issue report

### Future Enhancements

1. **Parallel processing**: Batch process multiple crashes
2. **Duplicate detection**: Check for similar existing issues
3. **Auto-bisect**: Find commit that introduced regression
4. **Coverage integration**: Track which code paths trigger bugs
5. **CVE workflow**: Extend for CVE submission

## Integration with Fuzzing Tools

### FeatureFuzz-SV Integration

FeatureFuzz-SV outputs crashes in the expected format:
- Output directory: `/edazz/FeatureFuzz-SV/output/crashes/`
- Directory naming: `assertion_<hash>_<timestamp>`
- Contains: `error.txt`, `source.sv`

**Batch processing script** (future):
```bash
for crash in /edazz/FeatureFuzz-SV/output/crashes/assertion_*; do
    python3 scripts/reproduce.py "$crash" && \
    python3 scripts/minimize.py ./circt-b* && \
    python3 scripts/generate_issue.py ./circt-b*
done
```

## Related Documentation

- Skill source: `.claude/skills/circt-bug-reporter/`
- Package: `.claude/skills/circt-bug-reporter.skill`
- Issue template reference: `.claude/skills/circt-bug-reporter/references/issue_template.md`
- CIRCT docs: https://circt.llvm.org/docs/
