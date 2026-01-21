# CIRCT Vulnerability Reproduction Environment

**CVE-PENDING** | CVSS 5.3 (Medium)  
**Issue:** [llvm/circt#9469](https://github.com/llvm/circt/issues/9469)  
**Fix PR:** [llvm/circt#9481](https://github.com/llvm/circt/pull/9481)

## 漏洞描述 (Vulnerability Description)

CIRCT 编译器在处理 SystemVerilog 中的直接数组索引（如 `clkin_data[0]`）作为时钟信号时存在不一致性。编译器无法处理这种有效的 SystemVerilog 代码模式，但使用中间线网赋值的语义等价代码可以成功编译。

The CIRCT compiler exhibits inconsistent handling of direct array indexing (e.g., `clkin_data[0]`) as clock signals in SystemVerilog `always_ff` sensitivity lists. The compiler fails to process this valid SystemVerilog pattern but succeeds with semantically equivalent code using intermediate wire assignments.

### 影响版本 (Affected Versions)
- CIRCT firtool-1.139.0 及更早版本 (and earlier)

### 影响组件 (Affected Components)
- `circt-verilog` (Frontend)
- `arcilator` (Backend)
- LLHD lowering pipeline (Mem2Reg, HoistSignals, Deseq passes)

## 快速开始 (Quick Start)

### 1. 构建 Docker 镜像 (Build Docker Image)

```bash
docker build -t circt-vuln-cve-pending .
```

### 2. 运行漏洞复现 (Run Vulnerability Reproduction)

**完整测试 (Full Test):**
```bash
docker run --rm circt-vuln-cve-pending
```

**仅测试漏洞代码 (Vulnerable Code Only):**
```bash
docker run --rm circt-vuln-cve-pending --vuln-only
```

**仅测试工作区代码 (Workaround Code Only):**
```bash
docker run --rm circt-vuln-cve-pending --workaround-only
```

**IR 分析 (IR Analysis):**
```bash
docker run --rm circt-vuln-cve-pending --analyze
```

### 3. 保存输出文件 (Save Output Files)

```bash
docker run --rm -v $(pwd)/results:/vuln-reproduction/output circt-vuln-cve-pending
```

输出文件将保存在 `./results/` 目录：
- `top1.err` - 漏洞代码错误输出
- `top2.out` - 工作区代码编译输出
- `top2.json` - 生成的状态文件
- `top1_detailed_ir.mlir` - 漏洞代码 IR 详细分析
- `top2_detailed_ir.mlir` - 工作区代码 IR 详细分析

## 文件说明 (File Descriptions)

### 测试用例 (Test Cases)

**top1.sv** - 漏洞代码 (Vulnerable Code)
```systemverilog
// 直接数组索引 - 编译失败
always_ff @(posedge clkin_data[0])
  if (!clkin_data[32]) _00_ <= 6'h00;
```
预期结果：编译失败，出现 `llhd.constant_time` 错误

**top2.sv** - 工作区代码 (Workaround Code)
```systemverilog
// 中间线网赋值 - 编译成功
wire clkin_0 = clkin_data[0];
wire rst = clkin_data[32];
always_ff @(posedge clkin_0)
  if (!rst) _00_ <= 6'h00;
```
预期结果：编译成功

### 核心文件 (Core Files)

- `Dockerfile` - 漏洞复现环境定义
- `reproduce.sh` - 自动化复现脚本
- `report.md` - 完整漏洞技术报告

## 预期输出 (Expected Output)

### 成功复现漏洞 (Successful Reproduction)

```
============================================================
  REPRODUCTION SUMMARY
============================================================

[VULNERABILITY CONFIRMED]

Status: VULNERABLE
Version: CIRCT firtool-1.139.0
Vulnerability: Direct array indexing in sensitivity lists causes
               compilation failure with llhd.constant_time error

Evidence:
  ✓ Vulnerable code (top1.sv) compilation FAILED
  ✓ Workaround code (top2.sv) compilation SUCCEEDED
  ✓ Error signature 'llhd.constant_time' detected

Impact:
  - Valid SystemVerilog code is rejected by compiler
  - Requires manual code restructuring
  - Affects automated hardware generation workflows

Recommendation:
  Apply patch from PR #9481
```

### 漏洞特征 (Vulnerability Signature)

```
error: failed to legalize operation 'llhd.constant_time' that was explicitly marked illegal
    %0 = llhd.constant_time <0ns, 0d, 1e>
         ^
```

## 手动测试 (Manual Testing)

### 进入容器 (Enter Container)

```bash
docker run --rm -it --entrypoint /bin/bash circt-vuln-cve-pending
```

### 手动编译测试 (Manual Compilation Test)

```bash
# 测试漏洞代码
circt-verilog --ir-hw top1.sv | arcilator --state-file=top1.json
# 预期：失败，llhd.constant_time 错误

# 测试工作区代码
circt-verilog --ir-hw top2.sv | arcilator --state-file=top2.json
# 预期：成功，生成 top2.json

# 查看详细 IR
circt-verilog --ir-hw --mlir-print-ir-before-all top1.sv > top1_ir.mlir 2>&1
```

## 修复建议 (Remediation)

### 临时工作区 (Immediate Workaround)

```systemverilog
// 使用中间线网提取数组元素
wire clk = clkin_data[0];
wire rst = clkin_data[32];

always_ff @(posedge clk)
  if (!rst) begin
    // 你的逻辑
  end
```

### 长期解决方案 (Long-term Solution)

升级到包含 PR #9481 修复的 CIRCT 版本：
```bash
# 从源码构建最新版本
git clone https://github.com/llvm/circt.git
cd circt
git checkout main  # 确保包含 PR #9481
# ... 按照官方文档构建
```

## 技术细节 (Technical Details)

### 根本原因 (Root Cause)

LLHD 降级管道中的 `Mem2Reg` 和 `HoistSignals` pass 无法识别数组元素访问（`a[0]`）作为有效的时钟信号候选，导致：

1. Frontend 未能正确标识 `clkin_data[0]` 为时钟信号
2. 降级 pass 无法将数组索引时钟信号转换为 `seq.firreg` 操作
3. 生成非法的 `llhd.constant_time` 操作
4. Arcilator 后端拒绝编译（明确标记 `llhd.constant_time` 为非法）

### 影响评估 (Impact Assessment)

| 类别 | 影响程度 | 说明 |
|------|---------|------|
| **设计正确性** | 中等 | 需要代码重构，但有简单工作区 |
| **工具互操作性** | 中等 | 影响某些自动化综合工具兼容性 |
| **开发工作流** | 中等 | 需要人工干预，但解决方案直接 |
| **安全性** | 低 | 编译时失败，非静默错误 |

### CVSS v3.1 评分 (Scoring)

**向量字符串:** `CVSS:3.1/AV:L/AC:L/PR:N/UI:R/S:U/C:N/I:L/A:L`

**基础分数:** 5.3 (Medium)

- **攻击向量 (AV):** Local - 需要本地访问编译设计
- **攻击复杂度 (AC):** Low - 标准 SystemVerilog 模式即可触发
- **用户交互 (UI):** Required - 用户必须尝试编译
- **完整性影响 (I):** Low - 需要代码修改，但有工作区
- **可用性影响 (A):** Low - 临时中断，有直接解决方案

## CWE 分类 (CWE Classification)

- **CWE-703:** Improper Check or Handling of Exceptional Conditions
- **CWE-697:** Incorrect Comparison (编译器未能识别等价信号表示)
- **CWE-1304:** Improperly Preserved Integrity of Hardware Configuration State

## 参考资料 (References)

- **GitHub Issue:** https://github.com/llvm/circt/issues/9469
- **Fix PR:** https://github.com/llvm/circt/pull/9481
- **CIRCT Documentation:** https://circt.llvm.org/
- **LLHD Dialect:** https://circt.llvm.org/docs/Dialects/LLHD/
- **完整技术报告:** `report.md`

## 致谢 (Acknowledgments)

- **发现者:** M2kar (@m2kar)
- **分析:** 5iri (@5iri)
- **维护者指导:** Fabian Schuiki (@fabianschuiki)
- **修复实现:** 5iri (@5iri)

## 许可 (License)

本复现环境和文档遵循 CIRCT 项目的许可证（Apache-2.0 with LLVM Exceptions）。

---

**文档版本:** 1.0  
**最后更新:** 2026-01-21  
**状态:** 公开披露，修复开发中
