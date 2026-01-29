# Bug Reporting Workflow - Output Files

## JSON Data Files (Intermediate Analysis)

| File | Size | Description |
|------|-------|-------------|
| `reproduce.json` | 2.5K | Reproduction verification data including crash signature and stack trace |
| `analysis.json` | 2.8K | Root cause analysis with hypotheses and keywords |
| `minimize.json` | 2.4K | Test case minimization process and reduction statistics |
| `validation.json` | 1.3K | Validation results and classification |
| `duplicates.json` | 2.3K | GitHub duplicate search results with similarity scores |
| `metadata.json` | 1.1K | Workflow metadata and tool information |
| `status.json` | 718B | Workflow status tracking |

## Markdown Reports

| File | Size | Description |
|------|-------|-------------|
| `root_cause.md` | 6.5K | Detailed root cause analysis report |
| `validation.md` | 2.7K | Validation report with tool compatibility checks |
| `duplicates.md` | 4.2K | Duplicate issue search report |
| `minimize_report.md` | 2.6K | Test case minimization report |
| `issue.md` | 5.8K | Final GitHub Issue report |

## Test Case Files

| File | Size | Description |
|------|-------|-------------|
| `source.sv` | 320B | Original SystemVerilog test case (22 lines) |
| `bug.sv` | 35B | Minimalized test case (2 lines) |

## Command and Log Files

| File | Size | Description |
|------|-------|-------------|
| `command.txt` | 41B | Simplified reproduction command |
| `error.txt` | 6.5K | Original crash log from fuzzing |
| `error.log` | 2.4K | Error log (copy of error.txt) |
| `reproduce.log` | 965B | Reproduction attempt output |

## Summary

**Total Files**: 17 files
**Total Size**: ~35 KB

**Workflow Status**: 
- ✅ Reproduce verification completed
- ✅ Root cause analysis completed  
- ✅ Test case minimization completed (90.9% reduction)
- ✅ Validation completed
- ✅ Duplicate checking completed
- ✅ Issue report generated

**Note**: Bug appears FIXED in current CIRCT version (LLVM 22.0.0git). Original crash was in CIRCT 1.139.0 arcilator LowerState pass.
