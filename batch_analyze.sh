#!/bin/bash

# ============================================================
# CIRCT Bug Reporter - Batch Analysis Script
# Description: Batch process all assertion_* directories using
#              OpenCode + circt-bug-reporter skill
# ============================================================

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CRASH_BASE_DIR="/edazz/crash-202060127"
LOG_DIR="./log"
WORK_BASE_DIR="$(pwd)"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Print banner
echo "============================================================"
echo "  CIRCT Bug Reporter - Batch Analysis"
echo "  OpenCode + /circt-bug-reporter skill"
echo "============================================================"
echo ""

# Count total directories
TOTAL_DIRS=$(ls -d "$CRASH_BASE_DIR"/assertion_* | wc -l)
echo -e "${BLUE}[INFO]${NC} Found $TOTAL_DIRS assertion directories to process"
echo ""

# Initialize counters
SUCCESS_COUNT=0
FAIL_COUNT=0
CURRENT=0

# Get all assertion directories
ASSERTION_DIRS=($(ls -d "$CRASH_BASE_DIR"/assertion_* | sort))

# Process each directory
for CRASH_DIR in "${ASSERTION_DIRS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    # Extract directory name (e.g., assertion_03ce98b35955_20260125_204541)
    DIR_NAME=$(basename "$CRASH_DIR")
    
    # Extract hash portion (e.g., 03ce98b35955)
    HASH=$(echo "$DIR_NAME" | grep -oP 'assertion_\K[0-9a-f]+(?=_)')
    
    # Work directory name
    WORK_DIR="circt-b${HASH}"
    LOG_FILE="$LOG_DIR/${WORK_DIR}.log"
    
    echo -e "${YELLOW}[${CURRENT}/${TOTAL_DIRS}]${NC} Processing: $DIR_NAME"
    echo -e "  Hash: $HASH"
    echo -e "  Work Dir: $WORK_DIR"
    echo -e "  Log File: $LOG_FILE"
    
    # Construct OpenCode command
    OPENCODE_CMD="opencode run --title \"$WORK_DIR\" --format json \"/ralph-loop /circt-bug-reporter $CRASH_DIR 。工作文件夹为 $WORK_DIR。 不要提交 issue! do not submit issue to github!\" < /dev/null >> \"$LOG_FILE\" 2>&1"
    
    echo -e "  Command: ${BLUE}$OPENCODE_CMD${NC}"
    
    # Execute command
    echo -e "${BLUE}[START]${NC} Starting analysis..."
    if eval "$OPENCODE_CMD"; then
        echo -e "${GREEN}[SUCCESS]${NC} Analysis completed for $WORK_DIR"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        EXIT_CODE=$?
        echo -e "${RED}[FAIL]${NC} Analysis failed for $WORK_DIR (exit code: $EXIT_CODE)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
    echo ""
done

# Print summary
echo "============================================================"
echo "  BATCH ANALYSIS SUMMARY"
echo "============================================================"
echo ""
echo "Total Processed: $TOTAL_DIRS"
echo -e "${GREEN}Success:${NC} $SUCCESS_COUNT"
echo -e "${RED}Failed:${NC} $FAIL_COUNT"
echo ""
echo "Log files location: $LOG_DIR/"
echo ""
echo "Next steps:"
echo "  1. Review log files for errors: ls -lh $LOG_DIR/"
echo "  2. Check generated work directories: ls -d circt-b*/"
echo "  3. Review generated reports: find circt-b* -name '*.md'"
echo ""
echo "============================================================"
