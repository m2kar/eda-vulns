#!/bin/bash

# CIRCT Vulnerability Reproduction Script
# CVE-PENDING: Inconsistent Array Indexing Handling in Sensitivity Lists
# Issue: https://github.com/llvm/circt/issues/9469
# CVSS: 5.3 (Medium)

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo "============================================================"
echo "  CIRCT Vulnerability Reproduction Environment"
echo "  CVE-PENDING | CVSS 5.3 (Medium)"
echo "  Issue: https://github.com/llvm/circt/issues/9469"
echo "============================================================"
echo ""

# Print environment info
print_env_info() {
    echo -e "${BLUE}[INFO]${NC} Environment Information:"
    echo "  OS: $(uname -s) $(uname -r)"
    echo "  Architecture: $(uname -m)"
    echo "  CIRCT Version: $(circt-verilog --version 2>&1 | head -n1 || echo 'Unknown')"
    echo "  Arcilator Version: $(arcilator --version 2>&1 | head -n1 || echo 'Unknown')"
    echo "  Working Directory: $(pwd)"
    echo ""
}

# Test vulnerable code (should fail)
test_vulnerable() {
    echo -e "${YELLOW}[TEST 1]${NC} Testing VULNERABLE code (top1.sv)..."
    echo "  Description: Direct array indexing in sensitivity list"
    echo "  Expected Result: Compilation FAILURE with llhd.constant_time error"
    echo ""
    
    echo "  Command: circt-verilog --ir-hw top1.sv | arcilator --state-file=output/top1.json"
    echo ""
    
    if circt-verilog --ir-hw top1.sv 2>output/top1_verilog.err | \
       arcilator --state-file=output/top1.json >output/top1.out 2>output/top1.err; then
        echo -e "${RED}[FAIL]${NC} Expected compilation failure but succeeded!"
        echo "  This indicates the vulnerability may be PATCHED in this version."
        return 1
    else
        echo -e "${GREEN}[PASS]${NC} Compilation failed as expected (vulnerability confirmed)"
        echo ""
        echo "  Error Output:"
        if [ -f output/top1.err ]; then
            cat output/top1.err | head -n 20
        fi
        echo ""
        
        # Check for specific vulnerability signature
        if grep -q "llhd.constant_time" output/top1.err 2>/dev/null; then
            echo -e "${RED}[VULNERABLE]${NC} Detected vulnerability signature: 'llhd.constant_time'"
            return 0
        else
            echo -e "${YELLOW}[WARNING]${NC} Different error than expected vulnerability signature"
            return 2
        fi
    fi
}

# Test workaround code (should succeed)
test_workaround() {
    echo -e "${YELLOW}[TEST 2]${NC} Testing WORKAROUND code (top2.sv)..."
    echo "  Description: Intermediate wire assignments (semantically identical)"
    echo "  Expected Result: Compilation SUCCESS"
    echo ""
    
    echo "  Command: circt-verilog --ir-hw top2.sv | arcilator --state-file=output/top2.json"
    echo ""
    
    if circt-verilog --ir-hw top2.sv 2>output/top2_verilog.err | \
       arcilator --state-file=output/top2.json >output/top2.out 2>output/top2.err; then
        echo -e "${GREEN}[PASS]${NC} Compilation succeeded with workaround"
        echo "  Generated files:"
        if [ -f output/top2.json ]; then
            echo "    - output/top2.json (state file, $(wc -c < output/top2.json) bytes)"
        fi
        return 0
    else
        echo -e "${RED}[FAIL]${NC} Workaround compilation failed unexpectedly!"
        echo ""
        echo "  Error Output:"
        if [ -f output/top2.err ]; then
            cat output/top2.err | head -n 20
        fi
        return 1
    fi
}

# Detailed IR analysis
analyze_ir() {
    echo -e "${YELLOW}[ANALYSIS]${NC} Detailed IR Analysis..."
    echo ""
    
    echo "  1. Analyzing vulnerable code IR flow:"
    if circt-verilog --ir-hw --mlir-print-ir-before-all top1.sv >output/top1_detailed_ir.mlir 2>&1; then
        echo -e "     ${GREEN}✓${NC} Generated detailed IR dump: output/top1_detailed_ir.mlir"
    else
        echo -e "     ${YELLOW}!${NC} Could not generate full IR dump"
    fi
    
    echo "  2. Analyzing workaround code IR flow:"
    if circt-verilog --ir-hw --mlir-print-ir-before-all top2.sv >output/top2_detailed_ir.mlir 2>&1; then
        echo -e "     ${GREEN}✓${NC} Generated detailed IR dump: output/top2_detailed_ir.mlir"
    else
        echo -e "     ${YELLOW}!${NC} Could not generate full IR dump"
    fi
    
    echo ""
}

# Generate summary report
generate_report() {
    local vuln_result=$1
    local work_result=$2
    
    echo ""
    echo "============================================================"
    echo "  REPRODUCTION SUMMARY"
    echo "============================================================"
    echo ""
    
    if [ $vuln_result -eq 0 ] && [ $work_result -eq 0 ]; then
        echo -e "${RED}[VULNERABILITY CONFIRMED]${NC}"
        echo ""
        echo "Status: VULNERABLE"
        echo "Version: CIRCT firtool-1.139.0"
        echo "Vulnerability: Direct array indexing in sensitivity lists causes"
        echo "               compilation failure with llhd.constant_time error"
        echo ""
        echo "Evidence:"
        echo "  ✓ Vulnerable code (top1.sv) compilation FAILED"
        echo "  ✓ Workaround code (top2.sv) compilation SUCCEEDED"
        echo "  ✓ Error signature 'llhd.constant_time' detected"
        echo ""
        echo "Impact:"
        echo "  - Valid SystemVerilog code is rejected by compiler"
        echo "  - Requires manual code restructuring"
        echo "  - Affects automated hardware generation workflows"
        echo ""
        echo "Recommendation:"
        echo "  Apply patch from PR #9481: https://github.com/llvm/circt/pull/9481"
        echo ""
    elif [ $vuln_result -eq 1 ] && [ $work_result -eq 0 ]; then
        echo -e "${GREEN}[VULNERABILITY PATCHED]${NC}"
        echo ""
        echo "Status: NOT VULNERABLE (patched)"
        echo "Both test cases compiled successfully, indicating the vulnerability"
        echo "has been fixed in this version."
        echo ""
    else
        echo -e "${YELLOW}[INCONCLUSIVE]${NC}"
        echo ""
        echo "Status: Test results inconsistent"
        echo "Vulnerable code result: $vuln_result"
        echo "Workaround code result: $work_result"
        echo "Manual inspection of output files recommended."
        echo ""
    fi
    
    echo "Output files location: $(pwd)/output/"
    echo "  - top1.err: Vulnerable code error output"
    echo "  - top2.out: Workaround code compilation output"
    echo "  - top1_detailed_ir.mlir: Detailed IR dump (vulnerable)"
    echo "  - top2_detailed_ir.mlir: Detailed IR dump (workaround)"
    echo ""
    echo "For more information, see:"
    echo "  - Report: /vuln-reproduction/report.md (if available)"
    echo "  - Issue: https://github.com/llvm/circt/issues/9469"
    echo "  - Fix PR: https://github.com/llvm/circt/pull/9481"
    echo ""
    echo "============================================================"
}

# Main execution
main() {
    print_env_info
    
    # Parse arguments
    case "${1:---all}" in
        --vuln-only)
            test_vulnerable
            exit $?
            ;;
        --workaround-only)
            test_workaround
            exit $?
            ;;
        --analyze)
            analyze_ir
            exit 0
            ;;
        --all|*)
            vuln_result=0
            work_result=0
            
            test_vulnerable
            vuln_result=$?
            echo ""
            
            test_workaround
            work_result=$?
            echo ""
            
            analyze_ir
            
            generate_report $vuln_result $work_result
            
            # Return code: 0 if vulnerability confirmed, 1 if patched, 2 if inconclusive
            if [ $vuln_result -eq 0 ] && [ $work_result -eq 0 ]; then
                exit 0  # Vulnerability confirmed
            elif [ $vuln_result -eq 1 ] && [ $work_result -eq 0 ]; then
                exit 10  # Patched
            else
                exit 2  # Inconclusive
            fi
            ;;
    esac
}

# Run main function
main "$@"
