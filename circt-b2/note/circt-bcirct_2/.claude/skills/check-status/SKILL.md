---
name: check-status
description: 检查 CIRCT Bug 报告工作流的当前状态。读取 status.json，验证文件完整性，支持中断恢复和状态修正。
argument-hint: (无参数，在当前目录操作)
allowed-tools: Shell(cat:*), Shell(ls:*), Shell(jq:*), Shell(test:*), Read, Write
---

# Skill: 状态检查

## 功能描述

检查当前工作目录的工作流状态，验证各步骤输出文件的完整性，支持：
1. 查看当前进度
2. 验证文件完整性
3. 诊断问题
4. 中断恢复建议
5. 状态修正

## 输入

当前目录（circt-b<id>/）

## 输出

- 状态报告（终端输出）
- 更新 `status.json`（如需修正）

## 状态定义

### 最终状态

| 状态码 | 描述 | 含义 |
|--------|------|------|
| `not_a_bug` | 不是 Bug | validate 判定为设计限制或无效测例 |
| `duplicate` | 重复 Issue | check-duplicates 发现高相似度 Issue |
| `reproduce_failed` | 复现失败 | reproduce 无法复现崩溃 |
| `report_ready` | 报告就绪 | issue.md 已生成，待人工审核 |
| `submitted` | 已提交 | Issue 已提交到 GitHub |

### 中间状态

| 状态码 | 描述 | 下一步骤 |
|--------|------|----------|
| `pending` | 待处理 | reproduce |
| `reproducing` | 复现中 | 完成 reproduce |
| `analyzing` | 分析中 | 完成 root-cause-analysis |
| `minimizing` | 最小化中 | 完成 minimize |
| `validating` | 验证中 | 完成 validate |
| `checking_duplicates` | 检查重复 | 完成 check-duplicates |
| `generating` | 生成中 | 完成 generate-issue |

## 执行步骤

### Step 1: 读取状态

```bash
echo "========================================"
echo "CIRCT Bug Reporter - Status Check"
echo "========================================"
echo ""
echo "Working directory: $(pwd)"
echo "Crash ID: $(basename $(pwd))"
echo ""

# 检查 status.json
if [ ! -f status.json ]; then
    echo "❌ status.json not found"
    echo ""
    echo "This directory has not been initialized."
    echo "Run the report-circt-bug skill to start."
    exit 1
fi

# 读取状态
CURRENT_STATUS=$(jq -r '.status' status.json)
CURRENT_PHASE=$(jq -r '.phase // "unknown"' status.json)
CREATED_AT=$(jq -r '.created_at // "unknown"' status.json)
UPDATED_AT=$(jq -r '.updated_at // "unknown"' status.json)

echo "Current Status: $CURRENT_STATUS"
echo "Phase: $CURRENT_PHASE"
echo "Created: $CREATED_AT"
echo "Last Updated: $UPDATED_AT"
echo ""
```

### Step 2: 验证文件完整性

```bash
echo "========================================"
echo "File Integrity Check"
echo "========================================"
echo ""

# 定义各步骤的预期文件
declare -A STEP_FILES
STEP_FILES[reproduce]="reproduce.log metadata.json"
STEP_FILES[root_cause_analysis]="root_cause.md analysis.json"
STEP_FILES[minimize]="bug.sv error.log command.txt"
STEP_FILES[validate]="validation.json validation.md"
STEP_FILES[check_duplicates]="duplicates.json duplicates.md"
STEP_FILES[generate_issue]="issue.md"

# 检查各步骤完成状态
check_step() {
    local step="$1"
    local files="${STEP_FILES[$step]}"
    local completed=$(jq -r ".steps.$step.completed // false" status.json)
    
    echo "Step: $step"
    echo "  Status in JSON: $completed"
    
    # 检查文件
    local all_exist=true
    for file in $files; do
        if [ -f "$file" ]; then
            echo "  ✓ $file exists"
        else
            echo "  ✗ $file missing"
            all_exist=false
        fi
    done
    
    # 判断实际状态
    if [ "$completed" = "true" ] && [ "$all_exist" = "false" ]; then
        echo "  ⚠️ INCONSISTENT: Marked complete but files missing"
        return 1
    elif [ "$completed" = "false" ] && [ "$all_exist" = "true" ]; then
        echo "  ⚠️ INCONSISTENT: Files exist but not marked complete"
        return 2
    elif [ "$completed" = "true" ] && [ "$all_exist" = "true" ]; then
        echo "  ✓ CONSISTENT: Complete"
        return 0
    else
        echo "  - CONSISTENT: Not complete"
        return 3
    fi
    
    echo ""
}

# 检查所有步骤
STEPS_ORDER="reproduce root_cause_analysis minimize validate check_duplicates generate_issue"
LAST_COMPLETE_STEP=""
FIRST_INCOMPLETE_STEP=""

for step in $STEPS_ORDER; do
    check_step "$step"
    result=$?
    
    if [ $result -eq 0 ]; then
        LAST_COMPLETE_STEP="$step"
    elif [ $result -eq 3 ] && [ -z "$FIRST_INCOMPLETE_STEP" ]; then
        FIRST_INCOMPLETE_STEP="$step"
    fi
    
    echo ""
done
```

### Step 3: 检查输入文件

```bash
echo "========================================"
echo "Input Files Check"
echo "========================================"
echo ""

# 原始输入（在 minimize 之前需要）
if [ -f source.sv ] || [ -f source.fir ] || [ -f source.mlir ]; then
    echo "✓ Source file exists"
    ls -la source.* 2>/dev/null
else
    # 检查是否已经最小化
    if [ -f bug.sv ] || [ -f bug.fir ] || [ -f bug.mlir ]; then
        echo "✓ Minimized test case exists (original source removed)"
    else
        echo "✗ No source file found"
    fi
fi

echo ""

if [ -f error.txt ]; then
    echo "✓ error.txt exists"
elif [ -f error.log ]; then
    echo "✓ error.log exists (minimized)"
else
    echo "✗ No error file found"
fi

echo ""
```

### Step 4: 诊断问题

```bash
echo "========================================"
echo "Diagnosis"
echo "========================================"
echo ""

diagnose_status() {
    case $CURRENT_STATUS in
        pending)
            echo "Status: Workflow not started"
            echo "Action: Run reproduce skill"
            ;;
        reproducing)
            if [ -f reproduce.log ]; then
                echo "Status: Reproduce completed but status not updated"
                echo "Action: Check reproduce.log and update status"
            else
                echo "Status: Reproduce in progress or failed"
                echo "Action: Run reproduce skill"
            fi
            ;;
        analyzing)
            if [ -f analysis.json ]; then
                echo "Status: Analysis completed but status not updated"
                echo "Action: Update status to 'minimizing'"
            else
                echo "Status: Analysis in progress or failed"
                echo "Action: Run root-cause-analysis skill"
            fi
            ;;
        minimizing)
            if [ -f bug.sv ] || [ -f bug.fir ] || [ -f bug.mlir ]; then
                echo "Status: Minimize completed but status not updated"
                echo "Action: Update status to 'validating'"
            else
                echo "Status: Minimize in progress or failed"
                echo "Action: Run minimize skill"
            fi
            ;;
        validating)
            if [ -f validation.json ]; then
                echo "Status: Validation completed but status not updated"
                echo "Action: Update status to 'checking_duplicates'"
            else
                echo "Status: Validation in progress or failed"
                echo "Action: Run validate skill"
            fi
            ;;
        checking_duplicates)
            if [ -f duplicates.json ]; then
                echo "Status: Duplicate check completed but status not updated"
                echo "Action: Update status to 'generating'"
            else
                echo "Status: Duplicate check in progress or failed"
                echo "Action: Run check-duplicates skill"
            fi
            ;;
        generating)
            if [ -f issue.md ]; then
                echo "Status: Issue generated but status not updated"
                echo "Action: Update status to 'report_ready'"
            else
                echo "Status: Issue generation in progress or failed"
                echo "Action: Run generate-issue skill"
            fi
            ;;
        report_ready)
            echo "Status: Report is ready for review"
            echo "Action: Review issue.md and submit manually or use gh CLI"
            ;;
        submitted)
            echo "Status: Issue has been submitted"
            echo "Action: No further action needed"
            ;;
        reproduce_failed)
            echo "Status: Bug could not be reproduced"
            echo "Action: Check if bug was fixed or try different version"
            ;;
        not_a_bug)
            echo "Status: Determined not to be a bug"
            echo "Action: Review validation report for details"
            ;;
        duplicate)
            echo "Status: Duplicate issue found"
            echo "Action: Check duplicates.md for related issues"
            ;;
        *)
            echo "Status: Unknown ($CURRENT_STATUS)"
            echo "Action: Manual investigation needed"
            ;;
    esac
}

diagnose_status
echo ""
```

### Step 5: 生成恢复建议

```bash
echo "========================================"
echo "Recovery Suggestions"
echo "========================================"
echo ""

# 如果是中间状态，建议下一步
case $CURRENT_STATUS in
    pending|reproducing)
        echo "To continue: Load /reproduce skill"
        ;;
    analyzing)
        echo "To continue: Load /root-cause-analysis skill"
        ;;
    minimizing)
        echo "To continue: Load /minimize skill"
        ;;
    validating)
        echo "To continue: Load /validate skill"
        ;;
    checking_duplicates)
        echo "To continue: Load /check-duplicates skill"
        ;;
    generating)
        echo "To continue: Load /generate-issue skill"
        ;;
    report_ready)
        echo "Review issue.md and submit:"
        echo "  gh issue create -R llvm/circt --title \"<title>\" --body-file issue.md"
        ;;
    *)
        echo "No action needed or workflow complete."
        ;;
esac

echo ""
```

### Step 6: 状态修正（可选）

```bash
# 如果需要修正状态
fix_status() {
    local new_status="$1"
    local reason="$2"
    
    echo "Fixing status: $CURRENT_STATUS -> $new_status"
    echo "Reason: $reason"
    
    jq --arg status "$new_status" \
       --arg timestamp "$(date -Iseconds)" \
       '.status = $status | .updated_at = $timestamp' \
       status.json > status.json.tmp && mv status.json.tmp status.json
    
    echo "Status updated."
}

# 自动修正明显的不一致
auto_fix() {
    # 如果文件存在但状态落后，自动推进
    if [ "$CURRENT_STATUS" = "generating" ] && [ -f issue.md ]; then
        fix_status "report_ready" "issue.md exists"
    elif [ "$CURRENT_STATUS" = "checking_duplicates" ] && [ -f duplicates.json ] && [ ! -f issue.md ]; then
        fix_status "generating" "duplicates check complete"
    elif [ "$CURRENT_STATUS" = "validating" ] && [ -f validation.json ] && [ ! -f duplicates.json ]; then
        fix_status "checking_duplicates" "validation complete"
    elif [ "$CURRENT_STATUS" = "minimizing" ] && [ -f bug.sv -o -f bug.fir -o -f bug.mlir ] && [ ! -f validation.json ]; then
        fix_status "validating" "minimize complete"
    elif [ "$CURRENT_STATUS" = "analyzing" ] && [ -f analysis.json ] && [ ! -f bug.sv ]; then
        fix_status "minimizing" "analysis complete"
    elif [ "$CURRENT_STATUS" = "reproducing" ] && [ -f reproduce.log ] && [ ! -f analysis.json ]; then
        fix_status "analyzing" "reproduce complete"
    fi
}

# 询问是否自动修正
echo "========================================"
echo "Auto-fix"
echo "========================================"
echo ""
echo "Checking for auto-fixable inconsistencies..."
# auto_fix  # 取消注释以启用自动修正
echo "Auto-fix disabled. To enable, edit the skill."
```

## 状态报告示例

### 正常进行中

```
========================================
CIRCT Bug Reporter - Status Check
========================================

Working directory: /home/user/circt-b1
Crash ID: circt-b1

Current Status: minimizing
Phase: in_progress
Created: 2026-01-27T14:00:00+00:00
Last Updated: 2026-01-27T14:30:00+00:00

========================================
File Integrity Check
========================================

Step: reproduce
  Status in JSON: true
  ✓ reproduce.log exists
  ✓ metadata.json exists
  ✓ CONSISTENT: Complete

Step: root_cause_analysis
  Status in JSON: true
  ✓ root_cause.md exists
  ✓ analysis.json exists
  ✓ CONSISTENT: Complete

Step: minimize
  Status in JSON: false
  ✗ bug.sv missing
  ✗ error.log missing
  ✗ command.txt missing
  - CONSISTENT: Not complete

========================================
Diagnosis
========================================

Status: Minimize in progress or failed
Action: Run minimize skill

========================================
Recovery Suggestions
========================================

To continue: Load /minimize skill
```

### 工作流完成

```
========================================
CIRCT Bug Reporter - Status Check
========================================

Current Status: report_ready
Phase: completed

All steps completed successfully.

========================================
Recovery Suggestions
========================================

Review issue.md and submit:
  gh issue create -R llvm/circt --title "<title>" --body-file issue.md
```

## 手动状态修正

如需手动修正状态：

```bash
# 设置为特定状态
jq '.status = "validating"' status.json > tmp.json && mv tmp.json status.json

# 标记步骤为完成
jq '.steps.reproduce.completed = true' status.json > tmp.json && mv tmp.json status.json

# 重置整个工作流
rm status.json
# 然后重新运行 report-circt-bug
```

## 强制重新开始

```bash
# 删除状态文件强制重新开始
rm -f status.json

# 或者删除所有生成文件
rm -f status.json reproduce.log metadata.json root_cause.md analysis.json \
      bug.sv error.log command.txt minimize_report.md \
      validation.json validation.md duplicates.json duplicates.md issue.md
```

## 注意事项

1. **状态一致性**：status.json 应该反映实际文件状态
2. **中断恢复**：可以从任意中间状态恢复
3. **自动修正**：谨慎使用自动修正功能
4. **手动修正**：必要时可以手动编辑 status.json
5. **备份**：修正前考虑备份 status.json
