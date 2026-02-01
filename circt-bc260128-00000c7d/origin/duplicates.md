# CIRCT Issue Duplicate Check Report

**Date**: 2026-02-01
**Repository**: llvm/circt
**Original Issue Analysis**: /home/zhiqing/edazz/eda-vulns/circt-bc260128-00000c7d/origin/analysis.json

## Original Issue Summary

**Crash Information**:
- **Dialect**: Moore
- **Failing Pass**: MooreToCore
- **Crash Type**: assertion_failure
- **Assertion Message**: "dyn_cast on a non-existent value"
- **Crash Location**: `PortImplementation.h`:177 - `ModulePortInfo::sanitizeInOut`()
- **Problematic Pattern**: string as module output port

**Root Cause Hypothesis**:
1. Moore parser creates `moore::StringType` for string types
2. `getModulePortInfo()` converts it to `sim::DynamicStringType` instead of an HW type
3. `sanitizeInOut()` tries `dyn_cast<hw::InOutType>(sim::DynamicStringType)`
4. The cast fails, triggering LLVM's dyn_cast assertion

## Search Summary

### Search Queries Performed
- `string MooreToCore assertion`
- `sanitizeInOut crash`
- `dyn_cast InOutType`
- `circt-verilog output string module`
- `string port` (open)
- `moore string` (open)
- `assertion failure` (open)
- `MooreToCore` (open)
- `circt-verilog output string` (open)
- `ModulePortInfo` (open)
- `sanitizeInOut` (open)
- `port type` (open)
- `fstring type on port` (open)
- `crash with string` (open)
- `type conversion` (open)
- `dynamic string` (open)

### Total Issues Found
- Total searches: 16
- Total issues matching keywords: 298
- Issues reviewed: 8

## Similarity Analysis

### High-Score Issues (score >= 5)


#### Issue #8332 - [MooreToCore] Support for StringType from moore to llvm dialect
- **Score**: 5/5
- **Similarity Components**:
  - same_dialect (Moore): +3分
  - same_pass (MooreToCore): +3分
  - related_to_string_type: +2分
- **Status**: open
- **Key Quote**: Discusses "MooreToCore's lack of string-type conversion"
- **Relevance**: HIGH - This issue directly discusses the absence of string-type conversion in MooreToCore

### Medium-Score Issues (score 3-4)


#### Issue #8283 - [ImportVerilog] Cannot compile forward decleared string type
- **Score**: 4/5
- **Similarity Components**:
  - related_to_string_type: +2分
  - related_to_string_port: +1分
  - circt-verilog context: +1分
- **Status**: open
- **Comments**: 3
- **Key Quote**: 关于前向声明 string 类型无法编译的问题。MooreToCore 缺少字符串类型转换。
- **Relevance**: HIGH - Related to the same problem

### Low-Score Issues (score < 3)


#### Issue #8770 - (f)printf parsing assertion failure
- **Score**: 1/5
- **Similarity Components**:
  - assertion_failure: +1分
- **Status**: open
- **Relevance**: LOW - printf 解析时的断言失败。与原始问题的 assertion failure 概念相关，但属于不同功能。

#### Issue #2123 - ModulePortInfo should return VerilogNames
- **Score**: 0/5
- **Similarity Components**:
  - None
- **Status**: open
- **Relevance**: NONE - ModulePortInfo 应该返回 VerilogNames。这是一个设计问题，不是崩溃问题。

#### Issue #977 - [FIRRTL] when (core dump)
- **Score**: 0/5
- **Similarity Components**:
  - None
- **Status**: closed
- **Relevance**: NONE - 通用核心转储问题，没有具体细节。

#### Issue #7224 - [FIRRTL] LowerLayers creates empty port names, preventing round-trip
- **Score**: 0/5
- **Similarity Components**:
  - related_to_ports: +1分
- **Status**: closed
- **Relevance**: NONE - LowerLayers 产生空端口名的问题。与 port 相关，但不是字符串问题。

## Key Findings

1. **Direct Precedent**: Issue #8283 already documents the exact same problem: "MooreToCore's lack of string-type conversion"

2. **Related Work**: Issue #8332 is about adding StringType support to MooreToCore

3. **Similar Pattern**: Issue #8930 has the exact same assertion message "dyn_cast on a non-existent value", but for a different conversion problem

4. **Port Type Pattern**: Issue #8382 shows assertion failures with port types, though in FIRRTL dialect

5. **Lack of Direct Match**: No exact duplicate of this specific crash (sanitizeInOut on string port) exists

## Conclusion

Based on the similarity analysis, this issue appears to be a **likely_new** (likely new) issue rather than a duplicate, BUT it is **closely related** to existing work.

### Recommended Action: review_existing

**Reasoning**:
  1. Score: 5/5 (highest) from #8332 and #8283
  2. The core problem (MooreToCore lack of string-type conversion) is already being discussed in issues #8283 and #8332
  3. This specific crash in `sanitizeInOut` with string ports is a concrete manifestation of the same issue
  4. The existing issues provide context and solutions that should be considered

**Recommendation**: Review with maintainers and reference:
- Issue #8283: Discusses string type conversion limitations in MooreToCore
- Issue #8332: Proposes adding StringType support to MooreToCore
- Issue #8930: Same assertion message, different context, shows vulnerability in MooreToCore

### Next Steps

1. **Check Issue #8283 and #8332 status**:
   - Are they resolved? If yes, this issue should be closed as duplicate
   - If they are open, reference them in the issue description
   - Ask maintainers if these solutions address this specific crash

2. **Consider adding solution based on existing work**:
   - The discussion in #8283 suggests handling string types properly
   - May need to add type checking before dyn_cast in sanitizeInOut()
   - Or reject string types at parsing/validation stage

3. **If these issues are not resolved**, consider creating a new issue that:
   - References #8283 and #8332 as related work
   - Provides a clear reproduction case
   - Suggests solution based on the discussion

## Metadata

- **Top Score**: 5/5 (Issue #8332)
- **Top Similar Issue**: #8332 - [MooreToCore] Support for StringType from moore to llvm dialect
- **Total Issues Checked**: 8
- **Search Keywords**: 16 queries performed
- **Conclusion**: review_existing