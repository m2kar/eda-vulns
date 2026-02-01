# CIRCT GitHub Issues 重复检查报告

## 崩溃概览

- **Crash ID**: bc260129-000014ab
- **Dialect**: LLHD
- **Crash Type**: assertion (IntegerType bitwidth limit)
- **Crash Location**: `lib/Dialect/LLHD/Transforms/Mem2Reg.cpp:1742` in `Promoter::insertBlockArgs(BlockEntry*)`

## 错误信息

```
error: integer bitwidth is limited to 16777215 bits
Assertion 'succeeded(ConcreteT::verifyInvariants(...))' failed in mlir::IntegerType::get()
```

## 搜索结果汇总

### 搜索1: "Mem2Reg"
- **结果数**: 2 个相关 Issue
- **最相关**: Issue #8693 - `[Mem2Reg] Local signal does not dominate final drive`
  - 相似度: 7.5/10
  - 共同点: 都是 Mem2Reg 通道中的 LLHD 信号处理问题
  - 差异: #8693 关注信号引用支配性，当前问题关注类型创建验证

### 搜索2: "LLHD bitwidth"
- **结果数**: 0 个直接匹配

### 搜索3: "integer assertion"
- **结果数**: 5 个Issue，都是低相似度匹配
  - Issue #8266: FIRRTL integer properties 相关
  - Issue #6405: FIRRTL 属性流检查
  - Issue #3235: Calyx pass 状态清理
  - Issue #3289: Python 绑定类型转换
  - 上述都与当前问题无关

## 相似度分析

| Issue# | Title | Score | 原因 |
|--------|-------|-------|------|
| 8693 | [Mem2Reg] Local signal does not dominate final drive | 7.5 | 同一 pass，同一方言，但不同的根本原因 |
| 8286 | [circt-verilog][llhd][arcilator] Verilog-to-LLVM lowering issues | 4.2 | 通用 LLHD 管道问题 |
| 8266 | [FIRRTL] Integer Property folders assert in getAPSInt | 2.1 | 仅 assertion 相似 |

## 问题特征

### 根本原因
```
当 Mem2Reg 通道处理 SystemVerilog 的 class handle 变量时：
1. hw::getBitWidth() 对 class/handle 类型返回无效值（可能为 -1）
2. 该值被直接用于创建 IntegerType
3. IntegerType::get() 验证失败，因为 bitwidth > 2^24-1 (16777215)
4. 导致 assertion 失败并崩溃
```

### 触发模式
```verilog
class my_class;
  // class body
endclass

always @(posedge clk) begin
  if (rst)
    my_class_var = some_value;  // <- 这里触发 Mem2Reg 处理 class handle
end
```

## 重复检查结论

### 推荐
**建议创建新 Issue** ✓

### 理由

1. **不同的根本原因**
   - Issue #8693: 信号引用的支配性违规
   - 当前问题: 类型创建时的 bitwidth 验证失败

2. **不同的崩溃位置**
   - Issue #8693: `insertBlockArgs()` 函数（父级调用）
   - 当前问题: `insertBlockArgs(BlockEntry*)` 函数（参数版本）

3. **不同的触发条件**
   - Issue #8693: 信号引用顺序问题
   - 当前问题: class/handle 类型的 bitwidth 计算问题

4. **独特的错误信息**
   - 明确指出 "integer bitwidth is limited to 16777215 bits"
   - 这是 MLIR IntegerType 的硬编码限制

5. **特定的类型问题**
   - class/handle 类型缺少类型验证
   - 需要在 `findPromotableSlots()` 中添加类型检查

## 修复建议

根据分析结果，需要：

1. ✅ 在 `findPromotableSlots()` 中添加类型验证，排除 class/handle 类型
2. ✅ 在 `insertBlockArgs()` 中验证 `hw::getBitWidth()` 返回值
3. ✅ 为不支持的类型发出正确的诊断，而不是崩溃

## 相关文件

- `lib/Dialect/LLHD/Transforms/Mem2Reg.cpp` (Line 1742, 1654, 764)
- `include/circt/Dialect/HW/HWTypes.h` (bitwidth 相关函数)
- `lib/Dialect/HW/HWOps.cpp` (hw::getBitWidth 实现)

---

**报告生成时间**: 2026-02-01
**检查方式**: GitHub Issues 搜索 + 相似度分析
**可信度**: 高 (High Confidence)
