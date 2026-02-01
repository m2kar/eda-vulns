# 最小化过程报告

## 原始测例

**文件**: source.sv  
**行数**: 13 行

```systemverilog
module top_module (
  inout wire my_pin
);

  logic [7:0] data_array [0:3];
  
  initial begin
    data_array <= '{default: 8'hFF};
  end
  
  assign my_pin = data_array[0][0] ? 1'bz : 1'b0;

endmodule
```

## 根因分析摘要

- **问题**: Arc 方言的 LowerState pass 不支持 `llhd::RefType`
- **崩溃点**: `LowerState.cpp:219`，`StateType::get()` 验证失败
- **根本原因**: `inout` 端口被转换为 `!llhd.ref<T>` 类型，但 `StateType` 要求已知的 bit width

## 最小化迭代过程

### 迭代 1: 原始测例验证
- **命令**: `circt-verilog --ir-hw source.sv | arcilator`
- **结果**: 崩溃，退出码 134 (SIGABRT)
- **错误**: `state type must have a known bit width; got '!llhd.ref<i1>'`

### 迭代 2: 仅保留 inout 端口
- **测试代码**: `module top(inout wire p); endmodule`
- **结果**: ✅ 崩溃复现，退出码 134
- **结论**: 找到最小触发条件

### 迭代 3: 验证是否可以进一步简化
- **尝试**: 去掉 `wire` 关键字 → 同样崩溃
- **结论**: `inout p` 是触发崩溃的最小核心

## 最终最小化测例

**文件**: bug.sv  
**行数**: 2 行

```systemverilog
module top(inout wire p);
endmodule
```

## 最小化统计

| 指标 | 值 |
|------|------|
| 原始行数 | 13 |
| 最小化行数 | 2 |
| 减少比例 | **84.6%** |

## IR 输出对比

**原始**:
```mlir
module {
  hw.module @top_module(in %my_pin : !llhd.ref<i1>) {
    hw.output
  }
}
```

**最小化**:
```mlir
module {
  hw.module @top(in %p : !llhd.ref<i1>) {
    hw.output
  }
}
```

两者都产生了 `!llhd.ref<i1>` 类型的端口，确认了问题的核心是 `inout` 端口的类型转换。

## 复现命令

```bash
export PATH=/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin:$PATH
circt-verilog --ir-hw bug.sv | arcilator
```

预期输出：SIGABRT (退出码 134)，包含错误 "state type must have a known bit width; got '!llhd.ref<i1>'"
