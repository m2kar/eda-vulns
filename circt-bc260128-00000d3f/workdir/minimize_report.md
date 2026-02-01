# 最小化过程报告

## 原始测例
- 文件: `testcase.sv`
- 行数: 20 行 (不含空行 19 行)
- 内容: 包含 `my_union` typedef、`Sub` 模块 (带 union 端口和逻辑)、`Top` 模块 (实例化 `Sub`)

## 最小化过程

### Step 1: 移除 Top 模块内部变量和复杂实例化
- 尝试简化 `Top` 模块只保留空实例化 `Sub s()`
- **结果**: 未崩溃 (警告: 端口未连接)
- **结论**: 单独的 `Sub` 模块声明可能足以触发崩溃

### Step 2: 移除 Top 模块，只保留带 union 端口的 Sub 模块
```systemverilog
typedef union packed {
  logic [31:0] a;
  logic [31:0] b;
} my_union;

module Sub(input my_union data_in);
endmodule
```
- **结果**: 崩溃复现 ✓

### Step 3: 简化为单字段 union
```systemverilog
typedef union packed {
  logic [31:0] a;
} my_union;

module Sub(input my_union data_in);
endmodule
```
- **结果**: 崩溃复现 ✓

### Step 4: 最小位宽 (logic a)
```systemverilog
typedef union packed { logic a; } u;
module m(input u i);
endmodule
```
- **结果**: 崩溃复现 ✓

### Step 5: 测试内联 union (无 typedef)
```systemverilog
module m(input union packed { logic a; } i);
endmodule
```
- **结果**: 崩溃复现 ✓

### Step 6: 测试 output 方向
```systemverilog
typedef union packed { logic a; } u;
module m(output u o);
endmodule
```
- **结果**: 崩溃复现 ✓

## 最终最小化测例
选择 Step 4 的形式，因为:
1. 保留 typedef 形式，代码可读性更好
2. 使用最简单的命名 (u, m, i)
3. 4 行代码，结构清晰

```systemverilog
// Minimal test case: packed union as module port crashes circt-verilog
typedef union packed { logic a; } u;
module m(input u i);
endmodule
```

## 最小化统计
- 原始行数: 20 行
- 最终行数: 4 行
- **减少比例: 80%**

## 关键发现
1. **根因确认**: packed union 类型作为模块端口是触发条件
2. **与端口方向无关**: input/output 均触发
3. **与 typedef 无关**: 内联定义同样触发
4. **与 union 字段数无关**: 单字段即可触发
5. **与位宽无关**: 1-bit 即可触发
6. **不需要实例化**: 单独模块声明即可触发

## 崩溃签名匹配
最小化测例与原始测例产生相同的崩溃栈:
- `SVModuleOpConversion::matchAndRewrite()` in MooreToCore.cpp
- `MooreToCorePass::runOnOperation()`

类型转换失败发生在将 Moore dialect 的 packed union 类型转换为 HW dialect 时。
