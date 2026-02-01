# Check-Duplicates Worker 工作总结

## 任务概览

- **工作ID**: check-duplicates
- **目标 Issue**: 260129-000018aa
- **执行时间**: 2026-02-01
- **状态**: ✅ 完成

## 工作流程

### 1️⃣ 参数提取 ✅

从 analysis.json 成功提取关键信息：

| 参数 | 值 |
|------|-----|
| **Crash Type** | timeout |
| **Dialect** | moore |
| **Key Constructs** | nested_modules, function_call_chain, always_comb_dependency |
| **Suspected Functions** | analyzeFanIn, convertModuleHeader, convertModuleBody, ProcedureOpConversion, ensureNoLoops |
| **Location Keywords** | ConvertToArcs, ImportVerilog, MooreToCore, SplitLoops |

### 2️⃣ GitHub 搜索 ✅

执行了 10 组关键词搜索：

| 搜索关键词 | 结果数 |
|---------|--------|
| timeout | 1 |
| ConvertToArcs | 4 |
| nested modules | 2 |
| moore dialect | 15 |
| always_comb | 10 |
| 其他关键词 | 更多 |
| **总计** | **29 个独特 Issue** |

### 3️⃣ 相似度计算 ✅

对所有 29 个 Issue 计算了相似度分数：

**相似度分布**:
- 🔴 高相似 (>6分): 0 个
- 🟡 中相似 (4-6分): 4 个
- 🟢 低相似 (<4分): 25 个

**相似度计分因素**:
- Crash Type 匹配: +3 分
- 关键构造匹配: +3 分
- 超时阶段匹配: +2 分
- 方言匹配: +1 分
- 时间接近: +1 分
- 函数/位置匹配: +0.5 分

### 4️⃣ 结果分析 ✅

#### Top 5 Most Similar Issues

| 排名 | Issue | 分数 | 匹配项 |
|------|-------|------|--------|
| 1 | #9570 | 5.5 | MooreToCore, function call chain, moore dialect |
| 2 | #8844 | 5.0 | always_comb, ImportVerilog, moore dialect |
| 3 | #8176 | 4.5 | always_comb, MooreToCore, moore dialect |
| 4 | #8211 | 4.5 | always_comb, MooreToCore, moore dialect |
| 5+ | ... | <3.5 | 低相似度 |

## 最终建议

### 📋 推荐结果

```
推荐: LIKELY_NEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ 最相似 Issue: #9570
✓ 最高相似度分数: 5.5/10 (中等相似)
✓ 潜在重复 Issues: 1 个 (>5分)
✓ 需要关注的 Issues: 4 个 (>4分)

原因: 发现 4 个中等相似度 Issue，但没有发现高度相似的 Issue。
      这个 Issue 可能是新的，但值得进一步审查。
```

### 🔍 需要审查的 Similar Issues

强烈建议在提交前审查以下 Issue：

1. **Issue #9570** - [Moore] Assertion in MooreToCore when module uses packed union type as port
   - 相似度: 5.5/10
   - 原因: 涉及 MooreToCore，function call chain，moore dialect
   - 链接: https://github.com/llvm/circt/issues/9570

2. **Issue #8844** - [circt-verilog] 'moore.case_eq' operand must be simple bit vector type
   - 相似度: 5.0/10
   - 原因: 涉及 always_comb，ImportVerilog，moore dialect
   - 链接: https://github.com/llvm/circt/issues/8844

3. **Issue #8176** - [MooreToCore] Crash when getting values to observe
   - 相似度: 4.5/10
   - 原因: 涉及 always_comb，MooreToCore，moore dialect
   - 链接: https://github.com/llvm/circt/issues/8176

## 输出文件

### 1. duplicates.json
- **大小**: 17 KB
- **内容**: 所有 29 个 Issue 的详细分析数据
- **包含**:
  - 每个 Issue 的相似度分数
  - 匹配项详细列表
  - 原始搜索参数
  - 创建日期和标签信息

### 2. duplicates.md
- **大小**: 6.0 KB
- **内容**: 人类可读的重复检查报告
- **包含**:
  - 检查概览和统计
  - 相似度分析结果
  - Top 5 Most Similar Issues
  - 详细分析和结论
  - 建议行动步骤

### 3. status.json (已更新)
- **内容**: 工作流程状态
- **更新**:
  - phase2 标记为 completed
  - 添加 check_duplicates 详细结果
  - phase 更新为 phase3_generate_issue

### 4. raw_search_results.json
- **内容**: GitHub 搜索的原始数据
- **包含**: 29 个 Issue 的完整 GitHub API 响应

## 关键发现

### 📌 Timeout Issue 特性

当前 Issue (260129-000018aa) 的特殊特点：
- ✅ 非常具体的组合条件: **3 级嵌套模块 + 函数调用链 + always_comb**
- ✅ 明确的错误类型: **编译超时（300 秒）**
- ✅ 清晰的触发点: **ConvertToArcs pass** 中的 analyzeFanIn 函数

### 🔎 搜索覆盖度

- 搜索了 10 个不同的关键词组合
- 发现了 29 个潜在相关的 Issue
- 未发现完全相同的 Issue（>6分）
- 发现了 4 个需要审查的相似 Issue（4-6分）

### ⚠️ 注意事项

1. **最相似的 Issue (#9570)** 涉及 MooreToCore assertion，不是 timeout
2. **Issue #8176 和 #8211** 也是 MooreToCore 相关，但是 crash 而非 timeout
3. 当前 Issue 的 **timeout + nested modules + function chain** 组合相对罕见

## 下一步建议

### ✅ 可立即执行

1. **再次确认相似 Issue**：
   - 检查 #9570, #8844, #8176, #8211 的详细信息
   - 确认这些 Issue 是否真的是重复

2. **收集更多信息** (如果是新 Issue):
   - 最小化测试用例: ✅ 已完成
   - 完整的错误日志: ✅ 已提供
   - 期望行为: ✅ 已说明

3. **提交 Issue** 前检查清单:
   - ✅ 标题清晰，包括模块信息
   - ✅ 使用适当标签: `[Moore]`, `[ConvertToArcs]`, `bug`
   - ✅ 包含最小化重现用例
   - ✅ 描述重现步骤
   - ✅ 提供错误日志

### 🎯 推荐提交标题

```
[Moore][ConvertToArcs] Compilation timeout on nested modules with function call chain in always_comb
```

### 🏷️ 推荐标签

- `[Moore]` - Moore dialect
- `[circt-verilog]` - circt-verilog 工具
- `[ConvertToArcs]` - 受影响的 pass
- `bug` - Issue 类型
- `timeout` - 特殊症状

## 统计汇总

| 指标 | 值 |
|------|-----|
| **检查耗时** | ~30 秒 |
| **搜索关键词** | 10 组 |
| **发现 Issues** | 29 个 |
| **高相似度** | 0 个 |
| **中相似度** | 4 个 |
| **低相似度** | 25 个 |
| **推荐** | likely_new |
| **最高分数** | 5.5/10 |

## 完成状态

```
✅ 参数提取
✅ GitHub 搜索
✅ 相似度计算
✅ 结果分析
✅ 报告生成
✅ Status 更新

总体: ✅ 完成
```

---

**生成时间**: 2026-02-01 13:07 UTC
**检查 ID**: 260129-000018aa
**推荐**: LIKELY_NEW (4-6分范围，需进一步审查)
