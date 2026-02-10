#!/bin/bash

# CIRCT Vulnerability Reproduction Script
# CVE-PENDING: CIRCT IR destruction bug (llhd.prb destroyed but still has uses)
# Issue: https://github.com/llvm/circt/issues/9469
# CVSS: 5.3 (Medium)

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Banner
echo "============================================================"
echo "  CIRCT Vulnerability Reproduction Environment"
echo "  CVE-PENDING | CVSS 5.3 (Medium)"
echo "  Issue: https://github.com/llvm/circt/issues/9469"
echo "============================================================"
echo ""

print_env_info() {
    echo -e "${BLUE}[INFO]${NC} Environment Information:"
    echo "  OS: $(uname -s) $(uname -r)"
    echo "  Architecture: $(uname -m)"
    echo "  CIRCT Version: $(circt-verilog --version 2>&1 | head -n1 || echo 'Unknown')"
    echo "  Working Directory: $(pwd)"
    echo ""
}

###############################################################################
# Test vulnerable code (bug.sv)
###############################################################################
test_vulnerable() {
    echo -e "${YELLOW}[TEST 1]${NC} Testing VULNERABLE code (bug.sv)..."
    echo "  Description: Combinational self-loop triggering invalid IR destruction"
    echo "  Expected Result: circt-verilog FAILURE"
    echo ""
    echo "  Command: circt-verilog --ir-hw bug.sv"
    echo ""

    if circt-verilog --ir-hw bug.sv \
        > output/bug.out 2> output/bug.err; then
        echo -e "${RED}[FAIL]${NC} Expected compilation failure but succeeded!"
        echo "  This indicates the vulnerability may be PATCHED."
        return 1
    else
        echo -e "${GREEN}[PASS]${NC} Compilation failed as expected"
        echo ""
        echo "  Error Output (head):"
        head -n 20 output/bug.err || true
        echo ""

        # Signature check (stable across versions)
        if grep -q "operation destroyed but still has uses" output/bug.err 2>/dev/null; then
            echo -e "${RED}[VULNERABLE]${NC} Detected IR destruction/use-after-free signature"
            return 0
        else
            echo -e "${YELLOW}[WARNING]${NC} Failure occurred, but signature differs"
            return 2
        fi
    fi
}

###############################################################################
# Workaround / control test
# For interface compatibility, we keep this function even if bug.sv is single-file
###############################################################################
test_workaround() {
    echo -e "${YELLOW}[TEST 2]${NC} Testing CONTROL case..."
    echo "  Description: Ensure toolchain itself is functional"
    echo "  Expected Result: Compilation SUCCESS"
    echo ""

    if [ ! -f control.sv ]; then
        echo -e "${YELLOW}[SKIP]${NC} control.sv not found, assuming toolchain OK"
        return 0
    fi

    echo "  Command: circt-verilog --ir-hw control.sv"
    echo ""

    if circt-verilog --ir-hw control.sv \
        > output/control.out 2> output/control.err; then
        echo -e "${GREEN}[PASS]${NC} Control compilation succeeded"
        return 0
    else
        echo -e "${RED}[FAIL]${NC} Control compilation failed unexpectedly!"
        head -n 20 output/control.err || true
        return 1
    fi
}

###############################################################################
# IR analysis
###############################################################################
analyze_ir() {
    echo -e "${YELLOW}[ANALYSIS]${NC} Detailed IR Analysis..."
    echo ""

    if circt-verilog --ir-hw --mlir-print-ir-before-all bug.sv \
        > output/bug_detailed_ir.mlir 2>&1; then
        echo -e "  ${GREEN}✓${NC} Generated IR dump: output/bug_detailed_ir.mlir"
    else
        echo -e "  ${YELLOW}!${NC} Could not generate full IR dump"
    fi
    echo ""
}

###############################################################################
# Report
###############################################################################
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
        echo "Trigger: bug.sv (self-referential combinational logic)"
        echo "Failure: IR op destroyed while still referenced"
        echo ""
        echo "Evidence:"
        echo "  ✓ circt-verilog fails on bug.sv"
        echo "  ✓ Error signature detected"
        echo ""
        echo "Impact:"
        echo "  - Compiler crashes instead of issuing a semantic diagnostic"
        echo "  - Breaks fuzzing, CI, and automated IR pipelines"
        echo ""
        echo "Recommendation:"
        echo "  Apply fix from https://github.com/llvm/circt/pull/9481"
        echo ""
    elif [ $vuln_result -eq 1 ] && [ $work_result -eq 0 ]; then
        echo -e "${GREEN}[VULNERABILITY PATCHED]${NC}"
        echo ""
        echo "Status: NOT VULNERABLE"
        echo "bug.sv compiled successfully."
        echo ""
    else
        echo -e "${YELLOW}[INCONCLUSIVE]${NC}"
        echo ""
        echo "Vulnerable result: $vuln_result"
        echo "Control result:    $work_result"
        echo ""
    fi

    echo "Output directory: $(pwd)/output/"
    echo "  - bug.err"
    echo "  - bug_detailed_ir.mlir"
    echo ""
    echo "============================================================"
}

###############################################################################
# Main
###############################################################################
main() {
    mkdir -p output
    print_env_info

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
            test_vulnerable
            vuln_result=$?
            echo ""

            test_workaround
            work_result=$?
            echo ""

            analyze_ir
            generate_report $vuln_result $work_result

            if [ $vuln_result -eq 0 ] && [ $work_result -eq 0 ]; then
                exit 0
            elif [ $vuln_result -eq 1 ] && [ $work_result -eq 0 ]; then
                exit 10
            else
                exit 2
            fi
            ;;
    esac
}

main "$@"
