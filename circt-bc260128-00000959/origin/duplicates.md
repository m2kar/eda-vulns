# CIRCT Bug #260128-00000959 重复检查报告

## 概述
检查是否存在与 `arcilator` 中 `sim.fmt.literal` legalization 失败相关的现有 GitHub Issue。

## 关键特征
- **错误类型**: Legalization failure
- **工具**: arcilator
- **操作**: sim.fmt.literal
- **方言**: sim dialect
- **触发模式**: SystemVerilog `$error()` 生成的格式字符串

## 搜索结果摘要

执行了 5 个搜索查询，共找到 6 个相关 Issue：

| Issue # | 分数 | 标题 | 相似度原因 |
|---------|------|------|----------|
| 9467 | 7.5 | arcilator fails to lower llhd.constant_time | 相同工具、相同错误类型、不同操作 |
| 8286 | 6.5 | circt-verilog][llhd][arcilator] Verilog-to-LLVM lowering issues | 相同工具、相关的降低/合法化 |
| 7692 | 5.5 | [Sim] Combine integer formatting ops into one op | sim 方言、格式化操作 |
| 8012 | 5.0 | [Moore][Arc][LLHD] Moore to LLVM lowering issues | 相关降低问题 |
| 6810 | 4.0 | [Arc] Add basic assertion support | 断言和错误处理相关 |
| 8817 | 3.5 | [FIRRTL] Support special substitutions in assert intrinsics | 断言和格式字符串相关 |

## 最接近的匹配

**Issue #9467**: [circt-verilog][arcilator] `arcilator` fails to lower `llhd.constant_time` generated from simple SV delay (`#1`)
- 相似度分数：7.5/10
- 链接：https://github.com/llvm/circt/issues/9467

### 为何相似
- ✅ 相同的工具：arcilator
- ✅ 相同的错误类型：legalization failure
- ❌ 不同的操作：llhd.constant_time vs sim.fmt.literal

虽然 Issue #9467 在 arcilator 中报告了 legalization 失败，但针对的是不同的操作。这表明 arcilator 存在更广泛的 legalization 能力缺陷。

## 结论

### 推荐：**likely_new**（可能是新问题）

虽然 Issue #9467 显示 arcilator 存在 legalization 缺陷，但没有找到专门针对 `sim.fmt.literal` 操作的现有 Issue。

### 关键发现
1. **无精确匹配**：没有现有 Issue 组合了 sim.fmt.literal + arcilator legalization failure
2. **相关的广泛问题**：Issue #9467 和 #8286 表明 arcilator 存在多个 legalization gap
3. **特定场景**：sim.fmt.literal 由 SystemVerilog `$error()` 生成，这是一个特定的触发场景

## 建议行动

1. ✅ **报告为新 Issue** - 提交新的 GitHub Issue，因为这是一个特定的、之前未报告的场景
2. 🔗 **交叉参考** - 在新 Issue 中参考 #9467 和 #8286，表明这是更广泛的 arcilator legalization 缺陷的一部分
3. 📋 **详细说明** - 强调 sim.fmt.literal 作为从 SystemVerilog $error() 生成的特定格式字符串操作的重要性

## 相似度计分规则应用

- 7.5 分用于 Issue #9467：错误类型匹配 ✓ + 工具匹配 ✓ + 不同操作 ✗
  - 完全相同(10.0) → 无
  - 高度相似(7.0-9.9) → #9467 (7.5)
  - 中等相似(4.0-6.9) → #8286 (6.5), #7692 (5.5), #8012 (5.0)
  - 低度相关(1.0-3.9) → #6810 (4.0), #8817 (3.5)
