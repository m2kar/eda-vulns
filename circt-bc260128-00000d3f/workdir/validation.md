# 测例验证报告

## 测例信息
- **文件**: `bug.sv`
- **内容**:
```systemverilog
// Minimal test case: packed union as module port crashes circt-verilog
typedef union packed { logic a; } u;
module m(input u i);
endmodule
```

## 验证结果

### 1. 语法合规性检查

| 工具 | 版本 | 命令 | 结果 |
|------|------|------|------|
| Verilator | 5.022 | `verilator --lint-only bug.sv` | ✅ Pass (无错误) |
| Slang | 10.0.6 | `slang --lint-only bug.sv` | ✅ Pass (0 errors, 0 warnings) |

**结论**: 测例是**合法的 SystemVerilog 代码**，符合 IEEE 1800-2017 标准。

### 2. 标准特性分析

- **使用特性**: packed union
- **标准参考**: IEEE 1800-2017 Section 7.3.1 (Packed unions)
- **说明**: Packed union 允许不同成员共享相同的存储空间，是 SystemVerilog 的标准特性
- **状态**: ✅ 合法标准特性

### 3. 跨工具验证

| 工具 | 行为 |
|------|------|
| Verilator 5.022 | 正常解析 |
| Slang 10.0.6 | 正常解析 |
| CIRCT circt-verilog (1.139.0) | ❌ 崩溃 |

### 4. 崩溃复现

**命令**:
```bash
/opt/firtool/bin/circt-verilog --ir-hw bug.sv
```

**崩溃类型**: Assertion Failure

**崩溃位置**: 
- `SVModuleOpConversion::matchAndRewrite()` in MooreToCore.cpp
- `MooreToCorePass::runOnOperation()`

**错误消息**: `dyn_cast on a non-existent value`

## 分类结论

### 分类: **Bug 报告 (report)**

### 理由:
1. ✅ 输入代码语法正确 (两个独立工具验证)
2. ✅ 使用标准 SystemVerilog 特性 (packed union)
3. ✅ 其他编译器可正常处理
4. ❌ CIRCT 崩溃而非报错

### Bug 性质:
- **类型**: 编译器内部错误 (Assertion Failure)
- **根因**: MooreToCore 转换未支持 packed union 类型的端口
- **影响**: 任何使用 packed union 作为模块端口的代码都会触发崩溃

## 建议
此 Bug 应报告给 CIRCT 项目，建议修复方向:
1. 在 MooreToCore TypeConverter 中添加 packed union 类型支持
2. 或在类型转换失败时给出友好错误消息而非崩溃
