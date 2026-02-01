# Validation Report

## Classification

**Result:** `report` (建议报告 Bug)

**Reason:** arcilator 缺少对 `sim.fmt.literal` 操作的 legalization 支持。合法的 SystemVerilog 立即断言 + `$error` 被 Verilator、Slang 和 circt-verilog 前端接受，但 arcilator 后端无法 lower 该操作。

## 测例信息

| 项目 | 值 |
|------|-----|
| 原始文件 | source.sv (22 行) |
| 最小化文件 | bug.sv (5 行) |
| 代码减少 | **77.3%** |

### 最小化测例 (bug.sv)

```systemverilog
module test;
  always @(*) begin
    assert (1'b0) else $error("fail");
  end
endmodule
```

### 复现命令

```bash
circt-verilog --ir-hw bug.sv | arcilator
```

## 跨工具验证

| 工具 | 版本 | 结果 | 说明 |
|------|------|------|------|
| Verilator | 5.022 | ✅ Pass | 无错误或警告 |
| Slang | 10.0.6 | ✅ Pass | 无错误或警告 |
| circt-verilog | - | ✅ Pass | 成功生成含 sim.fmt.literal 的 IR |
| arcilator | - | ❌ Fail | `failed to legalize operation 'sim.fmt.literal'` |

## 结论

测例语法合规，被多个工业级 SystemVerilog 工具接受。arcilator 无法处理 `sim.fmt.literal` 是一个实现缺陷，应作为 Bug 报告。
