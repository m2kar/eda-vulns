# CIRCT 漏洞 CVE 提交包
# CIRCT Vulnerability CVE Submission Package

**漏洞编号 / Vulnerability ID:** CVE-PENDING  
**CVSS 评分 / CVSS Score:** 5.3 (Medium)  
**发现日期 / Discovery Date:** 2026-01-18  
**发现者 / Discoverer:** M2kar (@m2kar)  
**GitHub Issue:** https://github.com/llvm/circt/issues/9469  
**Fix PR:** https://github.com/llvm/circt/pull/9481

---

## 📋 项目结构 / Project Structure

```
circt-1/
├── Dockerfile                  # Docker 漏洞复现环境定义
├── .dockerignore              # Docker 构建配置
├── report.md                  # 完整漏洞技术报告 (15KB)
├── README_DOCKER.md           # Docker 环境使用文档 (7.1KB)
├── TEST_RESULTS.md            # 实际测试结果报告 (4.9KB)
├── test.sh                    # 快速测试脚本
├── reproduce.sh               # 自动化复现脚本 (7.9KB)
├── top1.sv                    # 漏洞触发代码 (780B)
├── top2.sv                    # 工作区代码 (764B)
└── results/                   # 测试输出文件夹
    ├── top1.err               # 漏洞代码错误输出
    ├── top1_detailed_ir.mlir  # 漏洞代码 IR 分析 (66KB)
    ├── top2.json              # 成功编译的状态文件 (471B)
    ├── top2_detailed_ir.mlir  # 工作区代码 IR 分析 (56KB)
    └── top2.out               # 成功编译输出
```

---

## 🎯 漏洞概述 / Vulnerability Overview

### 中文描述

CIRCT 编译器在处理 SystemVerilog 敏感列表中的直接数组索引时存在不一致性。当使用 `clkin_data[0]` 作为时钟信号时，编译器无法正确处理，生成非法的 `llhd.constant_time` 操作导致编译失败。但使用语义等价的中间线网赋值方式可以成功编译。

**影响版本：** CIRCT firtool-1.139.0 及更早版本  
**影响组件：** circt-verilog, arcilator, LLHD lowering pipeline

### English Description

An inconsistency has been identified in CIRCT's handling of direct array indexing (e.g., `clkin_data[0]`) in SystemVerilog `always_ff` sensitivity lists. The compiler fails with an illegal `llhd.constant_time` operation error, but semantically equivalent code using intermediate wire assignments compiles successfully.

**Affected Versions:** CIRCT firtool-1.139.0 and earlier  
**Affected Components:** circt-verilog, arcilator, LLHD lowering pipeline

---

## 🚀 快速开始 / Quick Start

### 方式 1: 使用快速脚本 / Using Quick Script

```bash
# 1. 构建镜像 (首次运行)
./test.sh build

# 2. 运行完整测试
./test.sh run

# 3. 保存输出文件
./test.sh save

# 4. 查看其他选项
./test.sh help
```

### 方式 2: 使用 Docker 命令 / Using Docker Commands

```bash
# 构建镜像
docker build --platform linux/amd64 -t circt-vuln-cve-pending .

# 运行测试
docker run --platform linux/amd64 --rm circt-vuln-cve-pending

# 保存输出
docker run --platform linux/amd64 --rm \
  -v $(pwd)/results:/vuln-reproduction/output \
  circt-vuln-cve-pending
```

### 方式 3: 手动测试 / Manual Testing

```bash
# 进入容器
docker run --platform linux/amd64 --rm -it \
  --entrypoint /bin/bash circt-vuln-cve-pending

# 容器内手动运行
circt-verilog --ir-hw top1.sv | arcilator --state-file=top1.json  # 失败
circt-verilog --ir-hw top2.sv | arcilator --state-file=top2.json  # 成功
```

---

## 📊 测试结果 / Test Results

### ✅ 漏洞确认 / Vulnerability Confirmed

**测试平台 / Platform:** macOS (Apple M3 Pro) + Docker (linux/amd64)  
**测试日期 / Date:** 2026-01-21

| 测试项 / Test | 预期 / Expected | 实际 / Actual | 状态 / Status |
|--------------|----------------|--------------|--------------|
| 漏洞代码 (top1.sv) | 编译失败 | 编译失败 ❌ | ✅ PASS |
| 工作区代码 (top2.sv) | 编译成功 | 编译成功 ✅ | ✅ PASS |
| 错误特征检测 | llhd.constant_time | 检测到 | ✅ PASS |

**完整测试报告:** 查看 `TEST_RESULTS.md`

---

## 📄 文档说明 / Documentation

### 核心文档 / Core Documents

1. **report.md** (15KB)
   - 完整的 CVE 提交技术报告
   - 包含 12 个主要章节
   - CVSS v3.1 评分详细分析
   - CWE 分类和安全影响评估
   - 适用于 CVE 提交

2. **README_DOCKER.md** (7.1KB)
   - Docker 环境使用指南
   - 中英文双语说明
   - 包含快速开始、手动测试、修复建议
   - 适用于技术人员复现漏洞

3. **TEST_RESULTS.md** (4.9KB)
   - 实际测试结果报告
   - 包含错误输出、IR 分析
   - 生成文件清单
   - 适用于验证漏洞存在

### 技术文件 / Technical Files

4. **top1.sv** - 漏洞触发代码
   ```systemverilog
   always_ff @(posedge clkin_data[0])  // ❌ 编译失败
   ```

5. **top2.sv** - 工作区代码
   ```systemverilog
   wire clkin_0 = clkin_data[0];       // ✅ 编译成功
   always_ff @(posedge clkin_0)
   ```

6. **reproduce.sh** - 自动化复现脚本
   - 彩色输出
   - 四种运行模式
   - 自动生成报告

---

## 🔍 漏洞详细信息 / Vulnerability Details

### 根本原因 / Root Cause

LLHD lowering pipeline 的 `Mem2Reg` 和 `HoistSignals` passes 无法识别数组元素访问 (`clkin_data[0]`) 作为有效的时钟信号，导致：

1. Frontend 未能正确标识为时钟信号
2. 降级过程无法转换为 `seq.firreg` 操作
3. 生成非法的 `llhd.constant_time` 操作
4. Arcilator 后端拒绝编译

### 错误特征 / Error Signature

```
error: failed to legalize operation 'llhd.constant_time' that was explicitly marked illegal
    %0 = llhd.constant_time <0ns, 1d, 0e>
         ^
```

### 影响范围 / Impact Scope

- ❌ 直接数组索引作为时钟/复位信号
- ❌ 自动化硬件生成工具（如 Yosys）
- ❌ 多时钟域设计中的索引时钟选择
- ✅ 使用中间线网的等价代码可以工作

---

## 🛠️ 修复方案 / Remediation

### 临时工作区 / Immediate Workaround

```systemverilog
// ❌ 会失败的代码
always_ff @(posedge clkin_data[0])
  if (!clkin_data[32]) begin
    // logic
  end

// ✅ 工作区代码
wire clk = clkin_data[0];
wire rst = clkin_data[32];
always_ff @(posedge clk)
  if (!rst) begin
    // logic
  end
```

### 长期解决方案 / Long-term Solution

升级到包含 PR #9481 修复的 CIRCT 版本：

```bash
git clone https://github.com/llvm/circt.git
cd circt
git checkout main  # 确保包含 PR #9481
# 按照官方文档构建
```

---

## 📈 CVSS v3.1 评分 / CVSS Scoring

**向量字符串 / Vector String:**  
`CVSS:3.1/AV:L/AC:L/PR:N/UI:R/S:U/C:N/I:L/A:L`

**基础分数 / Base Score:** 5.3 (MEDIUM)

| 指标 / Metric | 值 / Value | 说明 / Rationale |
|--------------|-----------|------------------|
| 攻击向量 (AV) | Local | 需要本地访问编译环境 |
| 攻击复杂度 (AC) | Low | 标准 SystemVerilog 代码即可触发 |
| 所需权限 (PR) | None | 任何编译用户均可触发 |
| 用户交互 (UI) | Required | 用户必须尝试编译 |
| 范围 (S) | Unchanged | 影响限于编译流程 |
| 机密性 (C) | None | 无信息泄露 |
| 完整性 (I) | Low | 需要代码修改但有工作区 |
| 可用性 (A) | Low | 临时中断但解决方案直接 |

---

## 🔗 相关链接 / References

### GitHub 资源

- **Issue:** https://github.com/llvm/circt/issues/9469
- **Fix PR:** https://github.com/llvm/circt/pull/9481
- **Related Issue:** https://github.com/llvm/circt/issues/9467

### 官方文档

- **CIRCT:** https://circt.llvm.org/
- **LLHD Dialect:** https://circt.llvm.org/docs/Dialects/LLHD/
- **Arcilator:** https://circt.llvm.org/docs/Dialects/Arc/RationaleArc/

### CWE 分类

- **CWE-703:** Improper Check or Handling of Exceptional Conditions
- **CWE-697:** Incorrect Comparison
- **CWE-1304:** Improperly Preserved Integrity of Hardware Configuration State

---

## 👥 贡献者 / Contributors

- **发现者 / Reporter:** M2kar (@m2kar)
- **分析 / Analysis:** 5iri (@5iri)
- **维护者 / Maintainer:** Fabian Schuiki (@fabianschuiki)
- **修复实现 / Fix:** 5iri (@5iri)

---

## 📝 CVE 提交清单 / CVE Submission Checklist

- [x] 完整技术报告 (report.md)
- [x] 漏洞复现环境 (Dockerfile)
- [x] 漏洞触发代码 (top1.sv)
- [x] 工作区代码 (top2.sv)
- [x] 自动化测试脚本 (reproduce.sh)
- [x] 实际测试结果 (TEST_RESULTS.md)
- [x] CVSS v3.1 评分
- [x] CWE 分类
- [x] 时间线记录
- [x] 修复方案文档
- [x] 使用说明 (README_DOCKER.md)

---

## 📞 联系方式 / Contact

**发现者 / Reporter:** M2kar  
**GitHub:** @m2kar  
**Issue Tracker:** https://github.com/llvm/circt/issues/9469

---

**文档版本 / Document Version:** 1.0  
**最后更新 / Last Updated:** 2026-01-21  
**状态 / Status:** 准备提交 CVE / Ready for CVE Submission
