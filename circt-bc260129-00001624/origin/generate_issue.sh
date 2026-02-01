#!/bin/bash

# Step 1: 读取分析数据
DIALECT=$(jq -r '.dialect // "unknown"' analysis.json 2>/dev/null)
FAILING_PASS=$(jq -r '.pass // "unknown"' analysis.json 2>/dev/null)
CRASH_TYPE=$(jq -r '.crash_type // "unknown"' analysis.json 2>/dev/null)
ASSERTION_MSG=$(jq -r '.crash_signature.assertion_message // ""' analysis.json 2>/dev/null)
TOOL_VERSION=$(jq -r '.tool // "unknown"' metadata.json 2>/dev/null)
KEYWORDS=$(jq -r '.trigger_construct.keywords[0:3][]?' analysis.json 2>/dev/null | tr '\n' ' ')

echo "Dialect: $DIALECT"
echo "Failing pass: $FAILING_PASS"
echo "Crash type: $CRASH_TYPE"
echo "Keywords: $KEYWORDS"

# Step 2: 生成标题
ISSUE_TITLE="[Moore] Assertion failure in MooreToCore when using packed union as module port"

# Step 3: 提取根因摘要
ROOT_CAUSE_SUMMARY="MooreToCore pass crashes when processing a SystemVerilog packed union type used as a module port. The typeConverter.convertType() returns null for moore::UnionType, causing a dyn_cast assertion failure in hw::ModulePortInfo::sanitizeInOut()."

# Step 4: 提取 stack trace
STACK_TRACE=$(grep -E 'circt::|mlir::|llvm::' error.log | head -20)

# Step 5: 获取相关 Issue
RELATED_ISSUES="- #8930: [MooreToCore] Crash with sqrt/floor (similar dyn_cast failure in MooreToCore)"

# Step 6: 生成 issue.md
cat > issue.md << 'EOFHEADER'
<!-- 
  CIRCT Bug Report
  Testcase ID: 260129-00001624
-->

EOFHEADER

echo "<!-- Title: $ISSUE_TITLE -->" >> issue.md
echo "" >> issue.md

cat >> issue.md << 'EOFDESC'
## Description

EOFDESC

echo "$ROOT_CAUSE_SUMMARY" >> issue.md

cat >> issue.md << 'EOFDETAILS'

**Crash Type**: assertion
**Dialect**: Moore
**Failing Pass**: MooreToCore

**Key Issue**: MooreToCore lacks type converter for `moore::UnionType` (packed union). When converting module ports, the type converter returns null Type, which then triggers an assertion failure when dyn_cast is called on it.

EOFDETAILS

# Steps to Reproduce
cat >> issue.md << 'EOFREPRO'
## Steps to Reproduce

1. Save test case below as \`test.sv\`
2. Run:
   \`\`\`bash
   circt-verilog --ir-hw test.sv
   \`\`\`

EOFREPRO

# Test Case
echo "## Test Case" >> issue.md
echo "" >> issue.md
echo "\`\`\`systemverilog" >> issue.md
cat bug.sv >> issue.md
echo "\`\`\`" >> issue.md
echo "" >> issue.md

# Error Output
echo "## Error Output" >> issue.md
echo "" >> issue.md
echo "\`\`\`" >> issue.md
grep -E '(Assertion|error:|fatal:)' error.log | head -10 >> issue.md
echo "\`\`\`" >> issue.md
echo "" >> issue.md

# Root Cause Analysis
cat >> issue.md << 'EOFRCA'
## Root Cause Analysis

### Missing Type Converter

The MooreToCore pass in CIRCT does not have a type converter registered for \`moore::UnionType\` (packed union). This causes:

1. In \`getModulePortInfo()\`, \`typeConverter.convertType(unionType)\` returns \`Type{}\` (null)
2. The null Type is passed to \`hw::ModulePortInfo::sanitizeInOut()\`
3. \`sanitizeInOut()\` attempts \`dyn_cast<hw::InOutType>(p.type)\`
4. The dyn_cast fails with assertion: "dyn_cast on a non-existent value"

### Crash Location

- **File**: \`include/circt/Dialect/HW/PortImplementation.h:177\`
- **Function**: \`hw::ModulePortInfo::sanitizeInOut\`
- **Assertion**: \`dyn_cast on a non-existent value\`

### Suggested Fix

Add a type converter for \`moore::UnionType\` in \`MooreToCore.cpp\`. Options:
1. Convert packed union to equivalent bit-width integer type (simplest)
2. Convert union members to \`hw::StructType\` fields (preserves structure)
3. Add native \`hw::UnionType\` support (most comprehensive)

EOFRCA

# Environment
cat >> issue.md << 'EOFENV'
## Environment

- **CIRCT Version**: CIRCT 1.139.0
- **OS**: Linux
- **Architecture**: x86_64

EOFENV

# Stack Trace (折叠)
if [ -n "$STACK_TRACE" ]; then
cat >> issue.md << 'EOFSTACK'
## Stack Trace

<details>
<summary>Click to expand stack trace</summary>

\`\`\`
EOFSTACK

echo "$STACK_TRACE" >> issue.md
cat >> issue.md << 'EOFSTACKEND'
\`\`\`

</details>

EOFSTACKEND
fi

# Related Issues
if [ -n "$RELATED_ISSUES" ]; then
cat >> issue.md << 'EOFREL'
## Related Issues

$RELATED_ISSUES

Note: This issue is related to #8930 (both involve missing type converters in MooreToCore causing dyn_cast failures), but the triggering constructs are different (packed union vs sqrt/floor).

EOFREL
fi

# Footer
cat >> issue.md << 'EOFFOOTER'
---
*This issue was generated from testcase ID 260129-00001624 found via fuzzing.*

**Labels**: `Moore`, `crash`, `found-by-fuzzing`
EOFFOOTER

echo "========================================"
echo "issue.md generated successfully"
echo "========================================"
echo ""
echo "Title: $ISSUE_TITLE"
echo ""
