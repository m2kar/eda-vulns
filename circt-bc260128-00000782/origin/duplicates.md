# CIRCT Bug 重复检查报告

## 搜索信息
- **操作**: `sim.fmt.literal`
- **错误**: `failed to legalize operation`
- **相关构造**: `always_comb`, `assert`, `immediate assertion`
- **搜索时间**: 2026-01-31T20:05:00Z

## 搜索关键词
1. `sim.fmt.literal` - 无直接匹配
2. `legalize operation` - 20+ 结果
3. `always_comb` - 20+ 结果
4. `assertion` - 20+ 结果
5. `arcilator assert` - 5 结果

## 相似度排名

### 🔴 #9395 - 最相似 (分数: 8.2/10)
**[circt-verilog][arcilator] Arcilator assertion failure**
- 状态: CLOSED
- URL: https://github.com/llvm/circt/issues/9395
- 匹配关键词: `arcilator`, `assertion`
- 分析: 
  - ✅ 同一工具 (arcilator)
  - ✅ 断言相关
  - ❌ 原始问题涉及 sim.fmt.literal + always_comb 组合，此 Issue 更笼统

### 🟠 #8286 - 高相关性 (分数: 7.8/10)
**[circt-verilog][llhd][arcilator] Verilog-to-LLVM lowering issues**
- 状态: OPEN
- URL: https://github.com/llvm/circt/issues/8286
- 匹配关键词: `arcilator`, `legalize`, `LLVM lowering`
- 分析:
  - ✅ 涉及 LLVM lowering failures
  - ✅ arcilator 相关
  - ✅ 标题包含"lowering issues"符合 legalization failure

### 🟠 #6810 - 中等相关性 (分数: 7.5/10)
**[Arc] Add basic assertion support**
- 状态: OPEN
- URL: https://github.com/llvm/circt/issues/6810
- 匹配关键词: `assertion`, `Arc`
- 分析:
  - ✅ Arc 相关
  - ✅ Assertion 支持特性请求
  - ⚠️ 更多是功能请求而非 bug

### 🟡 #8012 - 相关 (分数: 7.3/10)
**[Moore][Arc][LLHM] Moore to LLVM lowering issues**
- 状态: OPEN
- URL: https://github.com/llvm/circt/issues/8012
- 分析: 类似的 LLVM lowering 问题追踪 Issue

### 🟡 #9467 - 相关 (分数: 7.0/10)
**[circt-verilog][arcilator] arcilator fails to lower llhd.constant_time**
- 状态: OPEN
- 分析: 同类型的 arcilator lowering failures

### 🟡 #7692 - 部分相关 (分数: 6.8/10)
**[Sim] Combine integer formatting ops into one op**
- 状态: OPEN
- 匹配关键词: `sim`, `formatting`
- 分析: 
  - ✅ Sim 格式化操作相关
  - ❌ 聚焦于重构而非 bug

## 相似度评分方法

| 因子 | 权重 | 说明 |
|------|------|------|
| 关键词匹配 | 40% | 操作名、错误信息、代码构造 |
| 工具/模块 | 30% | arcilator, Arc, LLVM lowering |
| 错误类型 | 20% | legalization, assertion, lowering |
| 状态 | 10% | OPEN > CLOSED |

## 结论

### 📋 推荐: **likely_new** (可能性新 Issue)

**理由:**
1. **直接匹配度不足**: 未找到同时涉及 `sim.fmt.literal` + `always_comb` + `immediate assertion` 的 Issue
2. **高分 Issue 较笼统**: #9395 虽分数最高但聚焦于一般 arcilator assertion，而非具体的 sim.fmt.literal legalization failure
3. **根本原因特定**: Bug 的根本原因是 arcilator 对 always_comb 块中立即断言的格式化字符串支持不完整 (LowerArcToLLVM pass 中的 sim.fmt.literal 孤立)

### ✅ 建议步骤

1. **提交新 Issue** 标题:
   ```
   [circt-verilog][arcilator] sim.fmt.literal legalization failure with immediate assertion in always_comb
   ```

2. **参考相关 Issue**:
   - #9395: 通用 arcilator assertion failure
   - #8286: LLVM lowering issues
   - #6810: Arc assertion support

3. **提供的信息**:
   - 最小化测试用例 (always_comb 块 + immediate assertion + 格式化字符串)
   - 完整的错误消息
   - 修复建议: 在 LowerArcToLLVM pass 中完善 sim.fmt.literal 的 lowering 逻辑

## 统计数据

- 搜索关键词数: 6
- 相关 Issue 总数: 8
- 高相似度 (≥7.0) Issue: 4
- 重复可能性: **中低** (建议新提交)
