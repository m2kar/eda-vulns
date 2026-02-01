# CIRCT 重复 Issue 检查报告

**Testcase ID**: 260129-0000175a  
**检查时间**: 2025-02-01  
**检查人**: check-duplicates-worker  

---

## 概述

对 CIRCT GitHub Issues 进行了全面搜索，旨在确定当前崩溃是否为已知问题的重复。

### 搜索策略

基于 `analysis.json` 和 `root_cause.md` 中提取的关键词，执行了以下搜索：

1. **"integer bitwidth"** - 寻找与位宽限制相关的问题
2. **"Mem2Reg"** - 寻找内存到寄存器提升 Pass 的相关问题
3. **"ClassHandleType"** - 寻找 Moore 类类型处理的问题
4. **"getBitWidth"** - 寻找位宽计算函数的相关问题

---

## 搜索结果汇总

### 搜索统计

| 搜索词 | 结果数 | 相关 Issue |
|------|-------|----------|
| integer bitwidth | 5 | 无直接相关 |
| Mem2Reg | 13 | 8693, 8286, 8494, 8245, 8246, 7483 等 |
| ClassHandleType | 0 | 无 |
| getBitWidth | 9 | 9287, 8930, 9269 等 |

### 发现的相关 Issue（按相似度排序）

---

## 🔴 顶级相关 Issue

### Issue #9287 ⭐⭐⭐ (相似度: 9.5/10)

**标题**: `[HW] Make hw::getBitWidth use std::optional vs -1`

**状态**: 🟢 OPEN

**链接**: https://github.com/llvm/circt/issues/9287

**描述**:
将 `circt::getBitWidth()` 转换为返回 `std::optional<uint64_t>` 而不是使用 -1 表示不支持的类型。

**为什么高度相关**:
- ✅ **直接根因**: 这个 Issue 正在计划修复我们崩溃的根本原因
- ✅ **位宽问题**: 当 `getBitWidth()` 返回 -1（不支持的类型）时，负值被隐式转换为极大的无符号整数
- ✅ **正在进行中**: 此 Issue 仍在开放状态，表明该问题未被完全解决
- ✅ **上游设计缺陷**: 所有依赖 `getBitWidth()` 的代码都潜在受影响

**关键评论**:
```
Convert circt::getBitWidth to return std::optional<uint64_t>. 
Also convert the BitWidthTypeInterface getBitWidth method to return 
the same instead of a signed version. Update the callsites. 
Where the callsites do not check for it, add an assertion.
```

**建议**: 
应作为主要参考 Issue 一起报告。这个 Issue 是长期解决方案，我们的 Testcase 是实际的崩溃示例。

---

### Issue #8245 (相似度: 8.0/10)

**标题**: `[LLHD] Mem2Reg crash on reasonable input`

**状态**: 🔴 CLOSED

**链接**: https://github.com/llvm/circt/issues/8245

**为什么相关**:
- ✅ 同样在 Mem2Reg Pass 中崩溃
- ✅ 同样涉及整数位宽问题
- ✅ 同样的崩溃位置 (Mem2Reg.cpp:1742)
- ❌ 不同的根因（与类类型无关）

**区别**:
虽然同样位置崩溃，但该 Issue 的 Testcase 不涉及 SystemVerilog 类，根因可能不同。

---

### Issue #7483 (相似度: 7.5/10)

**标题**: `[Moore] Mem2Reg Error`

**状态**: 🔴 CLOSED

**链接**: https://github.com/llvm/circt/issues/7483

**为什么相关**:
- ✅ 涉及 Mem2Reg Pass 处理 Moore 类型
- ✅ 涉及类型不匹配错误
- ✅ 与 SystemVerilog 类有关
- ❌ 具体错误类型不同（type mismatch vs integer bitwidth）

**区别**:
处理的是类型不匹配而非位宽溢出，但都反映 Mem2Reg 对 Moore 类型的支持不足。

---

## 其他相关 Issue

### Issue #8246 (相似度: 6.5/10)
- **标题**: `[LLHD] Mem2Reg creates drives to read-only signals`
- **状态**: CLOSED
- **相关性**: Mem2Reg 的另一个 Bug，同一 Pass

### Issue #8494 (相似度: 6.0/10)
- **标题**: `[LLHD] Mem2Reg does not properly combine enables of successive drives`
- **状态**: CLOSED
- **相关性**: Mem2Reg 的逻辑 Bug

### Issue #8693 (相似度: 5.5/10)
- **标题**: `[Mem2Reg] Local signal does not dominate final drive`
- **状态**: OPEN
- **相关性**: 开放的 Mem2Reg Bug，表明该 Pass 有多个问题

### Issue #8930 (相似度: 5.0/10)
- **标题**: `[MooreToCore] Crash with sqrt/floor`
- **状态**: OPEN
- **相关性**: 不同 Pass，但同样调用 `getBitWidth()` 导致 IntegerType 崩溃

---

## 分析总结

### 根因链

```
SystemVerilog 自引用类型
  ↓
Moore 的 ClassHandleType
  ↓
hw::getBitWidth() 返回 -1
  ↓
-1 转换为 unsigned (0xFFFFFFFFFFFFFFFF)
  ↓
IntegerType::get() 失败 (位宽 > 16777215)
  ↓
Mem2Reg.cpp:1742 断言失败 ❌
```

### 唯一性评估

| 方面 | 评分 | 说明 |
|-----|-----|------|
| Testcase 唯一性 | ⭐⭐⭐⭐⭐ | 自引用 typedef 的特定组合是新的 |
| 根因新颖性 | ⭐⭐ | 根因（getBitWidth 返回 -1）已知，见 #9287 |
| 触发条件新颖性 | ⭐⭐⭐⭐⭐ | SystemVerilog 类 + Mem2Reg 的特定组合 |
| 价值 | ⭐⭐⭐⭐⭐ | 为 #9287 提供新的具体测试用例 |

**结论**: 虽然根因已在 Issue #9287 中识别，但这个 Testcase 提供了一个**新的、具体的触发条件**，有助于验证修复。

---

## 推荐

### 最终建议: `review_existing` + 补充到 #9287

**理由**:
1. ✅ Issue #9287 直接解决根本原因
2. ✅ 我们的 Testcase 是该 Issue 的具体触发示例
3. ✅ 应作为测试用例附加到 #9287
4. ⚠️ 不应作为独立的新 Issue 报告
5. ⚠️ 应考虑作为 #8245 和 #7483 的补充参考

### 行动项

- [ ] 在 #9287 上评论，附加此 Testcase 作为 `getBitWidth() 返回 -1` 问题的具体示例
- [ ] 在 #7483 上评论，指出这是 Mem2Reg 无法处理 Moore 类型的另一个案例
- [ ] 建议将此 Testcase 添加到测试套件以防止回归

---

## 详细比对表

| 方面 | 当前 Crash | Issue #9287 | Issue #8245 | Issue #7483 |
|-----|---------|----------|---------|----------|
| **Pass** | Mem2Reg | (通用) | Mem2Reg | Mem2Reg |
| **方言** | LLHD (Moore) | HW | LLHD | Moore |
| **根因** | getBitWidth 返 -1 | 同左 | 不同 | Type mismatch |
| **错误位置** | Mem2Reg.cpp:1742 | 多个 | Mem2Reg.cpp:1742 | 不同 |
| **类型** | ClassHandleType | 通用 | i32 | l1 |
| **特殊性** | 自引用 typedef | 计划修复 | 基础场景 | 基础 Moore |
| **状态** | 新发现 | OPEN | CLOSED | CLOSED |

---

## 搜索日志

### 使用的命令

```bash
# 搜索整数位宽问题
gh search issues --repo llvm/circt "integer bitwidth" --limit 50

# 搜索 Mem2Reg 相关问题
gh search issues --repo llvm/circt "Mem2Reg" --limit 50

# 搜索 getBitWidth 相关问题
gh search issues --repo llvm/circt "getBitWidth" --limit 20

# 获取具体 Issue 详情
gh issue view <NUMBER> --repo llvm/circt --json title,body,state,labels,number
```

### 发现概览

- **总搜索查询**: 4
- **相关 Issue 数**: 11
- **最高相似度**: 9.5/10 (#9287)
- **确定有重复**: 是 (#9287 是根本原因)
- **可作为新 Issue 报告**: 否，应补充到 #9287

---

## 联系信息

**检查工作流**: check-duplicates  
**验证方式**: gh CLI with GitHub authentication  
**gh 版本**: 获取时已验证认证状态  

