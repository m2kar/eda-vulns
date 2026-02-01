# CIRCT Bug 重复检查报告

**生成时间**: 2026-02-01 12:32:24  
**分析ID**: 260129-00001855

---

## 📋 Bug 摘要

| 项目 | 内容 |
|------|------|
| **Dialect** | arc |
| **工具** | arcilator |
| **Pass** | LowerStatePass |
| **错误信息** | `state type must have a known bit width; got '!llhd.ref<i1>'` |
| **关键词** | `arcilator`, `LowerState`, `StateType`, `llhd.ref`, `inout`, `bidirectional port`, `computeLLVMBitWidth`, `assertion failure` |

---

## 🔍 搜索策略

### 使用的查询

- Query 1: `arcilator`
- Query 2: `LowerState`
- Query 3: `StateType`
- Query 4: `llhd.ref`
- Query 5: `inout`
- Query 6: `bidirectional port`
- Query 7: `computeLLVMBitWidth`
- Query 8: `assertion failure`

### 搜索结果统计

- **总查询数**: 6
- **找到的Issues**: 3
- **分析的Issues**: 3

---

## 🎯 重复检查结果

### 🚨 建议: **LIKELY_DUPLICATE**

**原因**: 找到高度相似的Issue #9574 (相似度: 90.0%)

### 匹配评分

| Issue # | 相似度 | 标题 | 状态 |
|---------|--------|------|------|
| #9574 | 90.0% | [Arc] Assertion failure when lowering inout ports in sequent... | OPEN |
| #6810 | 10.0% | [Arc] Add basic assertion support | OPEN |
| #8825 | 10.0% | [LLHD] Switch from hw.inout to a custom signal reference typ... | OPEN |


---

## 📊 详细分析结果

### 最相似的Issue: #9574

**相似度**: 90.0%


**标题**: [Arc] Assertion failure when lowering inout ports in sequential logic

**URL**: [https://github.com/llvm/circt/issues/9574](https://github.com/llvm/circt/issues/9574)

**状态**: OPEN

#### 相似度评分详解

- **keywords**: 75.0%
- **error_message**: 100.0%
- **tool_dialect**: 100.0%
- **pass**: 100.0%
- **sequence**: 1.6%

#### 匹配详情

- **匹配的关键词**: `arcilator`, `LowerState`, `StateType`, `llhd.ref`, `inout`, `assertion failure`
- **错误信息匹配**: ✅ 是
- **工具匹配**: ✅ 是
- **Dialect匹配**: ✅ 是
- **Pass匹配**: ✅ 是

---

### 所有匹配的Issues


#### 1. Issue #9574 - 相似度 90.0%

**标题**: [Arc] Assertion failure when lowering inout ports in sequential logic

**链接**: https://github.com/llvm/circt/issues/9574

**状态**: OPEN

**匹配的关键词**:
- `arcilator`
- `LowerState`
- `StateType`
- `llhd.ref`
- `inout`
- `assertion failure`

#### 2. Issue #6810 - 相似度 10.0%

**标题**: [Arc] Add basic assertion support

**链接**: https://github.com/llvm/circt/issues/6810

**状态**: OPEN

**匹配的关键词**:
- 无

#### 3. Issue #8825 - 相似度 10.0%

**标题**: [LLHD] Switch from hw.inout to a custom signal reference type

**链接**: https://github.com/llvm/circt/issues/8825

**状态**: OPEN

**匹配的关键词**:
- `llhd.ref`
- `inout`

---

## 💡 建议

### ⚠️ 可能是重复报告

此Bug与 Issue #9574 高度相似 (相似度 90.0%)。

**建议操作**:
1. 审查 Issue #9574 的内容
2. 如果确认是同一问题，可以关闭此Bug或添加参考链接
3. 如果是不同的问题，请更新Issue描述以明确差异

**参考链接**: https://github.com/llvm/circt/issues/9574


---

## 📈 搜索查询总结

使用的搜索查询:

- `repo:llvm/circt arcilator LowerState`
- `repo:llvm/circt StateType llhd.ref`
- `repo:llvm/circt inout port arc`
- `repo:llvm/circt arcilator assertion`
- `repo:llvm/circt LowerStatePass`
- `repo:llvm/circt llhd.ref type`


---

## 🔧 技术细节

### Bug 特征

**Pass**: LowerStatePass

**Dialect**: arc

**工具**: arcilator

**错误类型**: assertion

**关键词**:
- `arcilator`
- `LowerState`
- `StateType`
- `llhd.ref`
- `inout`
- `bidirectional port`
- `computeLLVMBitWidth`
- `assertion failure`


### 根本原因

computeLLVMBitWidth() in ArcTypes.cpp does not handle llhd::RefType, causing StateType verification to fail for bidirectional (inout) ports

**缺失的处理器**: llhd::RefType in computeLLVMBitWidth()

**不支持的类型**: !llhd.ref<i1>

### 触发构造

**类型**: inout_port

**SystemVerilog**: `inout logic port_a`

**IR类型**: `!llhd.ref<i1>`

---

## 📝 注意事项

- 相似度分数基于关键词匹配 (40%)、错误信息匹配 (30%)、工具/Dialect匹配 (20%) 和Pass匹配 (10%)
- 搜索结果基于GitHub Issues API的可用数据
- 建议始终进行人工审查以确认重复关系
- 如果Issue已在llvm/circt中存在，可以添加+1反应或新增信息

---

**生成者**: CIRCT Bug 重复检查系统  
**版本**: 1.0  
**最后更新**: 2026-02-01 12:32:24
