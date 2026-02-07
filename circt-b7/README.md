# [circt-verilog][LLHD] Assigning a signal to itself makes Sig2Reg hang

**Vulnerability ID:** CVE-PENDING  
**CVSS Score:** 6.1 (Medium)  
**Discovery Date:** 2026-02-01  
**Discoverer:**  M2kar (@m2kar) kaituo-crypto(@kaituo-crypto)  
**GitHub Issue:** https://github.com/llvm/circt/issues/9576  
**Fix PR:** https://github.com/llvm/circt/pull/9590


---

## Table of Contents

- [1. Vulnerability Overview](#1-vulnerability-overview)
- [2. Technical Details](#2-technical-details)
- [3. Reproduction](#3-reproduction)
- [4. Impact Analysis](#4-impact-analysis)
- [5. Remediation](#5-remediation)
- [6. CVE Classification](#6-cve-classification)
- [7. References](#7-references)

---

## 1. Vulnerability Overview

### 1.1 Description

CIRCT compiler hangs indefinitely when processing invalid SystemVerilog code that has both procedural (non-blocking) and continuous assignments to the same signal. Other tools like Verilator and Icarus Verilog correctly reject this code with an error message. The compiler should detect this pattern and emit a diagnostic instead of hanging.

### 1.2 Affected Scope

- **Affected Versions:** CIRCT firtool-1.139.0 and earlier
- **Affected Components:** CIRCT StstemVerilog and LLHD Sig2Reg Pass
- **Affected Scenarios:**
  - ❌ Code with both procedural and continuous assignments to the same signal (e.g., `q <= d` and `assign q = q`) causes infinite recursion during LLHD lowering.
  - In the main branch (commit e4838c703), the compiler crashes with a stack dump instead. This leaves uncertainty about whether the hang has been fixed or merely replaced with a different failure mode.

---

## 2. Technical Details

### 2.1 Root Cause

The compiler encounters multiple drivers to the same signal (`q`) during the Sig2Reg pass in LLHD lowering:

- Procedural driver: `always @(negedge clk) q <= d`
- Continuous driver: `assign q = q` (creates a combinational loop)

This causes infinite recursion in the signal promotion logic, resulting in a hang. Later main branch commits may produce a crash instead of a hang, but the underlying issue remains unhandled invalid code.

### 2.2 Error Signature

**Behavior on affected versions:**

```
$ circt-verilog --ir-hw bug.sv
# Hangs indefinitely (no output, no error)
# manually killed after 60+ seconds
```

**Main branch (commit e4838c703):**

```
$ circt-verilog --ir-hw bug.sv
...
LLVM ERROR: operation destroyed but still has uses
PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
Stack dump: ...
```

### 2.3 Vulnerable Code Example

```systemverilog
module bug(input logic clk, output logic q);
  logic d;

  always @(negedge clk) begin
    q <= d;  // Non-blocking procedural assignment
  end

  assign q = q;  // Continuous assignment to same signal
endmodule
```

---

## 3. Reproduction

### 3.1 Project Structure

```
circt-b7/
├── bug.sv                    # Vulnerable SystemVerilog code
├── reproduce.sh              # Script to reproduce hang
├── test.sh                   # Script to run docker
├── Dockerfile                # Configuration to the reproduce environment
├──error.log                  # output of CIRCT firtool-1.139.0
├──error.e4838c703.log        # output of main branch e4838c703
└──README.md                  # report of the vulnerability
```

### 3.2 Quick Start

```bash
# 1. Build image (first run)
./test.sh build

# 2. Run full test
./test.sh run
```
### 3.3 Cross-Tool Validation

| Tool | Behavior |
|------|----------|
| Icarus Verilog | ✅ Correctly rejects with error |
| Verilator | ✅ Correctly rejects with error |
| CIRCT | ❌ Hangs indefinitely (or crashes on main branch) |

---

## 4. Impact Analysis

### 4.1 Functional Impact

| Category | Level | Description |
|----------|-------|-------------|
| Design Correctness | 🟡 MEDIUM | Code must be refactored to avoid compiler hang |
| Tool Interoperability | 🟡 MEDIUM | Automation and synthesis tools may hang |
| Development Workflow | 🟡 MEDIUM | Manual intervention required to continue compilation |
| Verification Coverage | 🟢 LOW | Workaround code preserves behavior |

### 4.2 Security Implications

- **Code Integrity Risk:** Manual modifications introduce human error
- **Supply Chain Vulnerability:** Automated pipelines may hang on invalid code
- **Compiler Trust:** Inconsistent handling reduces reliability

### 4.3 Affected Use Cases

- FPGA development with multi-driver signals
- ASIC designs with procedural + continuous assignments
- Hardware fuzzing pipelines
- Legacy IP migration

---

## 5. Remediation



Upgrade to CIRCT version that includes PR #9590 fix:

```bash
git clone https://github.com/llvm/circt.git
cd circt
git checkout main  # Ensure fix included
# Build per official instructions
```



---

## 6. CVE Classification

### 6.1 CVSS v3.1 Scoring

`CVSS:3.1/AV:L/AC:L/PR:N/UI:R/S:U/C:N/I:L/A:H`  
Base Score: 6.1 (MEDIUM)

| Metric | Value | Rationale |
|--------|-------|-----------|
| Attack Vector (AV)| Local (L)|Requires local access to compilation environment |
| Attack Complexity(AC) | Low (L)| Standard SystemVerilog triggers hang |
| Privileges Required (PR) | None (N)| Any compilation user can trigger |
| User Interaction (UI) | Required (R)| User must try to compile vulnerable code |
| Scope(S) | Unchanged (U)| Impact limited to compilation execution |
| Confidentiality(C) | None (N) | No disclosure |
| Integrity(I) | Low (L)| Requires code modification |
| Availability(A) | High (H) | Causes the compiler to hang indefinitely, resulting in a denial of service |

### 6.2 CWE Classification

- **CWE-835:**  Infinite Loop / Denial of Service

### 6.3 Risk Assessment

**Risk Level:**  CVSS 6.1  
**Recommended Priority:** 🟡 MEDIUM

---

## 7. References

### 7.1 GitHub Resources

- **Issue:** https://github.com/llvm/circt/issues/9576
- **Fix PR:** (https://github.com/llvm/circt/pull/9590)

### 7.2 Official Documentation

- **CIRCT:** https://circt.llvm.org/
- **LLHD Dialect:** https://circt.llvm.org/docs/Dialects/LLHD/

### 7.3 Contributors

- **Reporter:** M2kar (@m2kar)  kaituo-crypto (@kaituo-crypto )
- **Analysis:** 5iri (@5iri)
- **Maintainer:** Fabian Schuiki (@fabianschuiki)
- **Fix Implementation:** 5iri (@5iri)

---

## 8. Appendix

### 8.1 Test Environment

- **OS:** Ubuntu 24.04 x86_64 in Docker
- **CIRCT Version:** firtool-1.139.0
- **LLVM Version:** 22.0.0git

### 8.2 Disclosure Policy

This report follows coordinated disclosure practices:

1. ✅ Public issue tracker used (GitHub)
2. ✅ Maintainers engaged, fix in progress
3. ✅ No active exploitation observed
4. ✅ Workaround available
5. ⏳ Awaiting CVE assignment and official advisory

**Status:** Public disclosure with active fix development

---

## 9. Contact

**Reporter:** M2kar  kaituo-crypto     
**GitHub:** [@m2kar](https://github.com/m2kar) [@kaituo-crypto](https://github.com/kaituo-crypto)  
**Email:** zhiqing.rui@gmail.com   kaituotenshi@gmail.com  
**Issue Tracker:** https://github.com/llvm/circt/issues/9576

---

**Document Version:** 1.0  
**Last Updated:** 2026-02-07  
**Status:** Ready for CVE Submission  
**License:** MIT

