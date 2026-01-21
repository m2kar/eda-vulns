# EDA Vulnerabilities Research

Collection of vulnerability research and CVE submissions for EDA (Electronic Design Automation) tools, focusing on hardware compilers, synthesis tools, and simulation platforms.

## Overview

This repository contains vulnerability reports, proof-of-concept code, and reproduction environments for security issues discovered in various EDA tools.

## Structure

```
eda-vulns/
├── circt-1/          # CIRCT vulnerability CVE-PENDING (CVSS 5.3)
│   ├── report.md     # Full technical CVE report
│   ├── Dockerfile    # Reproduction environment
│   └── ...           # Test files and scripts
└── ...               # Future vulnerability reports
```

## Current Vulnerabilities

### 1. CIRCT - Array Indexing in Sensitivity Lists (CVE-PENDING)

**Status:** 🔴 VULNERABLE  
**CVSS Score:** 5.3 (MEDIUM)  
**Affected Version:** CIRCT firtool-1.139.0 and earlier  
**Discovery Date:** 2026-01-18  
**Discoverer:** M2kar (@m2kar)

**Description:**  
CIRCT compiler fails to handle direct array indexing (e.g., `clkin_data[0]`) in SystemVerilog `always_ff` sensitivity lists, generating illegal `llhd.constant_time` operations.

**Links:**
- [GitHub Issue](https://github.com/llvm/circt/issues/9469)
- [Fix PR](https://github.com/llvm/circt/pull/9481)
- [Full Report](circt-1/report.md)
- [Docker Environment](circt-1/README_DOCKER.md)

**Quick Start:**
```bash
cd circt-1
./test.sh build    # Build Docker image
./test.sh run      # Run vulnerability reproduction
```

## Repository Guidelines

### Vulnerability Report Format

Each vulnerability should include:

1. **Technical Report** (`report.md`)
   - Executive summary
   - Vulnerability description
   - Proof of concept
   - Impact analysis
   - CVSS scoring
   - CWE classification
   - Timeline
   - References

2. **Reproduction Environment** (Docker preferred)
   - Dockerfile
   - Test cases (vulnerable + workaround)
   - Automated test scripts
   - Usage documentation

3. **Test Results**
   - Actual test outputs
   - IR/intermediate analysis
   - Verification evidence

### Directory Naming Convention

```
<tool-name>-<sequence-number>/
```

Examples:
- `circt-1/`
- `yosys-1/`
- `verilator-1/`

## Disclosure Policy

This repository follows coordinated vulnerability disclosure:

1. ✅ Report to vendor via preferred channel (usually GitHub Issues)
2. ✅ Allow reasonable time for vendor to develop fix
3. ✅ Coordinate disclosure timing with vendor
4. ✅ Publish after fix is available or disclosure deadline
5. ✅ Submit CVE request with full documentation

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Before submitting:**
   - Verify the vulnerability with vendor
   - Ensure fix is in progress or available
   - Prepare complete documentation

2. **Submission checklist:**
   - [ ] Full technical report
   - [ ] Proof of concept code
   - [ ] Reproduction environment
   - [ ] CVSS score calculation
   - [ ] CWE classification
   - [ ] Vendor communication evidence
   - [ ] Fix PR or patch (if available)

3. **Pull request:**
   - Create directory following naming convention
   - Include all required files
   - Update this README with vulnerability summary

## Research Focus

This repository focuses on:

- **Compiler bugs** in hardware synthesis tools
- **Logic synthesis inconsistencies**
- **Simulation mismatches**
- **Tool chain vulnerabilities**
- **HDL language implementation issues**

**Out of scope:**
- Design bugs in specific hardware projects
- General software vulnerabilities unrelated to EDA
- Theoretical attacks without practical impact

## Tools Covered

- [CIRCT](https://circt.llvm.org/) - Circuit IR Compilers and Tools
- [Yosys](https://yosyshq.net/yosys/) - Open-source synthesis tool
- [Verilator](https://www.veripool.org/verilator/) - Verilog simulator
- Other open-source EDA tools

## Contact

**Maintainer:** M2kar (@m2kar)  
**Email:** zhiqing.rui@gmail.com

For security-sensitive reports, please use vendor's preferred disclosure channel first.

## License

Documentation and reports: CC BY 4.0  
Code and scripts: MIT License

See individual directories for specific licenses.

---

**Disclaimer:** This research is conducted for improving security of EDA tools. All vulnerabilities are responsibly disclosed to vendors before publication.
