# Duplicate Check Report

## Search Queries
- `string module port`
- `MooreToCore`
- `InOutType`
- `dyn_cast`

## Search Results Summary
- **Total Issues Found**: 23
- **Analyzed**: 23 issues across 4 search queries
- **Scoring System**: Title keywords (2.0), Body keywords (1.0), Assertion match (3.0), Dialect label (1.5)

---

## Top Matches

### 🔴 Issue #8283: [ImportVerilog] Cannot compile forward decleared string type
**URL**: https://github.com/llvm/circt/issues/8283  
**Similarity Score**: 8.5 / 10  
**State**: OPEN  
**Is Duplicate**: **Yes - Very Likely**

**Matched Keywords**:
- `string` (title match: +2.0)
- `MooreToCore` (+1.5)
- `StringType` (+1.0)
- `type conversion` (+1.0)
- `moore.variable` (+1.0)
- `string type` (+2.0)

**Summary**:
This issue reports the **exact same root cause**: MooreToCore's lack of string-type conversion. The error message is:
```
error: failed to legalize operation 'moore.variable'
string str;
```

While our crash manifests as an assertion failure in `getModulePortInfo()` and this issue shows a legalization failure in `moore.variable`, **both stem from the same underlying problem**: no registered type converter for `moore::StringType` in MooreToCore.

**Key Similarities**:
- Both use SystemVerilog `string` type
- Both fail during MooreToCore pass
- Root cause: Missing StringType conversion

**Difference**: Our test case uses string as module input port, while #8283 uses string as local variable.

---

### 🟡 Issue #8332: [MooreToCore] Support for StringType from moore to llvm dialect
**URL**: https://github.com/llvm/circt/issues/8332  
**Similarity Score**: 7.5 / 10  
**State**: OPEN  
**Is Duplicate**: No (Feature Discussion)

**Matched Keywords**:
- `string` (title: +2.0)
- `MooreToCore` (title: +1.5)
- `StringType` (+1.0)
- `type conversion` (+1.0)
- `VariableOp` (+1.0)
- `moore` (+1.0)

**Summary**:
This is a **feature request/discussion** about adding StringType support to MooreToCore. The author proposes an implementation approach using sim dialect. While related to our bug, this is not a bug report but a feature proposal.

---

### 🟡 Issue #8930: [MooreToCore] Crash with sqrt/floor
**URL**: https://github.com/llvm/circt/issues/8930  
**Similarity Score**: 6.5 / 10  
**State**: OPEN  
**Is Duplicate**: No (Different Trigger)

**Matched Keywords**:
- `MooreToCore` (title: +1.5)
- `"dyn_cast on a non-existent value"` (assertion match: +3.0)
- `Moore` label (+1.5)
- `type conversion` (+0.5)

**Summary**:
This issue has the **identical assertion message**: `"dyn_cast on a non-existent value"`. However, it's triggered by `real` type conversion (sqrt/floor operations), not string type. This demonstrates a pattern: MooreToCore crashes when type conversion returns null for unsupported types.

**Insight**: Both bugs point to the same defensive programming issue - MooreToCore doesn't validate type conversion results.

---

### 🟢 Issue #7535: [MooreToCore] VariableOp lowered failed
**URL**: https://github.com/llvm/circt/issues/7535  
**Similarity Score**: 5.0 / 10  
**State**: OPEN  
**Is Duplicate**: No (Different Type)

**Matched Keywords**:
- `MooreToCore` (title: +1.5)
- `InOutType` (+1.5)
- `type conversion` (+1.0)
- `VariableOp` (+1.0)

**Summary**:
Crash when casting `hw::InOutType` for struct type lowering. Related type conversion issue but for **struct type**, not string type.

---

### 🟢 Issue #8973: [MooreToCore] Lowering to math.ipow?
**URL**: https://github.com/llvm/circt/issues/8973  
**Similarity Score**: 2.0 / 10  
**State**: OPEN  
**Is Duplicate**: No

**Summary**:
Discussion about lowering PowUOp in MooreToCore. Unrelated to string type issue.

---

## Recommendation

### **Verdict: `review_existing`**
### **Confidence: High**

### Recommended Action:
1. **Issue #8283 is a strong candidate duplicate** - Both report the same underlying problem (missing StringType conversion in MooreToCore)
2. Consider adding a comment to #8283 with this new test case (string as module port)
3. Alternatively, file a new issue but link to #8283 as related

### Root Cause Pattern Identified:
Multiple issues (#8283, #8930, our crash) show that **MooreToCore doesn't properly handle unconvertible types**. The fix should:
1. Add type converters for unsupported types (StringType, RealType, etc.)
2. Add null-check defensive validation after `typeConverter.convertType()`

---

## Keywords Used for Search
`string`, `module port`, `InOutType`, `MooreToCore`, `dyn_cast`, `StringType`, `type conversion`, `getModulePortInfo`, `sanitizeInOut`, `DynamicStringType`
