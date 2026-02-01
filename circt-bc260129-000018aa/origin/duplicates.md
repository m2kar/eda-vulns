# CIRCT 重复 Issue 检查报告

## 检查概览

- **检查时间**: 2026-02-01T13:06:56.501064
- **Target Issue**: 260129-000018aa
- **Crash Type**: timeout
- **Dialect**: moore
- **搜索 Issues 总数**: 29

## 搜索参数

### 关键特征
- **Crash Type**: timeout
- **Dialect**: moore
- **Key Constructs**: nested_modules, function_call_chain, always_comb_dependency

### 搜索关键词

#### 地点关键词
- ImportVerilog
- MooreToCore
- SplitLoops

#### 函数关键词
- analyzeFanIn
- Converter::analyzeFanIn
- convertModuleHeader
- Context::convertModuleHeader
- convertModuleBody
（还有 5 个更多）

## 相似度分析结果

### 📊 汇总统计

| 相似度等级 | 数量 | 阈值 |
|---------|------|------|
| 高相似 (review_existing) | 0 | > 6 分 |
| 中相似 (likely_new) | 4 | 4-6 分 |
| 低相似 (new_issue) | 25 | < 4 分 |

### 🎯 最终建议

**Recommendation**: `LIKELY_NEW`


#### 原因
发现 4 个中等相似度 Issue，这个 Issue 可能是新的，但值得进一步审查。


### 🏆 Top 5 Most Similar Issues


#### 1. Issue #9570 - 5.5 分
- **标题**: [Moore] Assertion in MooreToCore when module uses packed union type as port
- **创建时间**: 2026-02-01
- **标签**: 无
- **匹配项**:
    - function call chain
  - MooreToCore pass involved
  - moore dialect
  - verilog-related
  - suspected function/location: matchAndRewrite
- **预览**: ## Description

CIRCT crashes with assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"` when compiling SystemVerilog modules that ...


#### 2. Issue #8844 - 5.0 分
- **标题**: [circt-verilog]  'moore.case_eq' operand must be simple bit vector type, but got array
- **创建时间**: 2025-08-12
- **标签**: ImportVerilog
- **匹配项**:
    - always_comb/combinational logic
  - ImportVerilog pass involved
  - moore dialect
  - verilog-related
- **预览**: I ran `circt-verilog` on this file from OpenTitan:
https://github.com/lowRISC/opentitan/blob/d8b5efd1427152b8387d6e03d9db413167e58475/hw/ip/lc_ctrl/rt...


#### 3. Issue #8176 - 4.5 分
- **标题**: [MooreToCore] Crash when getting values to observe
- **创建时间**: 2025-02-03
- **标签**: Moore
- **匹配项**:
    - always_comb/combinational logic
  - MooreToCore pass involved
  - moore dialect
- **预览**: The following crashes due to an unattached region when calling `getValuesToObserve`.

```mlir
moore.module @crash(in %in0: !moore.i32, in %in1: !moore...


#### 4. Issue #8211 - 4.5 分
- **标题**: [MooreToCore]Unexpected observed values in llhd.wait.
- **创建时间**: 2025-02-08
- **标签**: Moore
- **匹配项**:
    - always_comb/combinational logic
  - MooreToCore pass involved
  - moore dialect
- **预览**: Please check this PR(https://github.com/llvm/circt/pull/8210/files) to view the details.

Or for example:
```
moore.module @crash(in %in0: !moore.i32,...


#### 5. Issue #7535 - 3.5 分
- **标题**: [MooreToCore] VariableOp lowered failed
- **创建时间**: 2024-08-20
- **标签**: 无
- **匹配项**:
    - MooreToCore pass involved
  - moore dialect
  - verilog-related
- **预览**: Dear @maerhart @fabianschuiki ,
 When lowering `SV` to `Hw` Dialect, there is a stack dump. 
Driver: circt-verilog %s
```
module top();
  typedef...


## 详细分析

### 高相似度 Issues (0)

无


### 中相似度 Issues (4)


##### Issue #9570: 5.5 分
- **标题**: [Moore] Assertion in MooreToCore when module uses packed union type as port
- **创建时间**: 2026-02-01
- **匹配项**: function call chain, MooreToCore pass involved, moore dialect, verilog-related, suspected function/location: matchAndRewrite


##### Issue #8844: 5.0 分
- **标题**: [circt-verilog]  'moore.case_eq' operand must be simple bit vector type, but got array
- **创建时间**: 2025-08-12
- **匹配项**: always_comb/combinational logic, ImportVerilog pass involved, moore dialect, verilog-related


##### Issue #8176: 4.5 分
- **标题**: [MooreToCore] Crash when getting values to observe
- **创建时间**: 2025-02-03
- **匹配项**: always_comb/combinational logic, MooreToCore pass involved, moore dialect


##### Issue #8211: 4.5 分
- **标题**: [MooreToCore]Unexpected observed values in llhd.wait.
- **创建时间**: 2025-02-08
- **匹配项**: always_comb/combinational logic, MooreToCore pass involved, moore dialect


## 结论

### 结果汇总


- **最相似 Issue**: #9570
- **最高相似度分数**: 5.5 分
- **潜在重复 Issues** (>5分): 1 个
- **需要关注的 Issues** (>4分): 4 个

### 建议行动


1. **可能是新 Issue**，但应审查以下相似 Issue：

   - Issue #9570: [Moore] Assertion in MooreToCore when module uses packed union type as port (5.5 分)
   - Issue #8844: [circt-verilog]  'moore.case_eq' operand must be simple bit vector type, but got array (5.0 分)
   - Issue #8176: [MooreToCore] Crash when getting values to observe (4.5 分)
2. 确保当前 Issue 包含：
   - 最小化的测试用例（已完成）
   - 清晰的错误重现步骤
   - 期望和实际行为的说明
3. 提交 Issue 前再次检查上述相似 Issues


---

## 附录

### 搜索统计

- **总搜索 Issues**: 29
- **高相似度** (>6分): 0
- **中相似度** (4-6分): 4
- **低相似度** (<4分): 25

### 相似度计算方法

相似度分数基于以下因素计算（总分 10 分）：

| 因素 | 分数 | 说明 |
|-----|------|------|
| Crash Type 匹配 (timeout) | +3 | Issue 描述中提到 timeout/hang/infinite loop |
| 关键构造匹配 | +3 | 匹配 nested modules、function calls、always_comb |
| 超时阶段匹配 | +2 | 涉及 ConvertToArcs、MooreToCore、SplitLoops 等 Pass |
| 方言匹配 (moore) | +1 | Issue 与 Moore 方言相关 |
| 时间接近 (6个月内) | +1 | 在最近 6 个月内创建 |
| 函数/位置匹配 | +0.5 | 提及可疑函数或位置 |

### 相似度阈值解释

- **新 Issue** (<4 分): 所有找到的 Issue 都几乎没有相关性
- **可能是新 Issue** (4-6 分): 有一些相关的 Issue，但不够相似
- **审查现有 Issue** (>6 分): 发现了非常相似的 Issue，很可能是重复的

