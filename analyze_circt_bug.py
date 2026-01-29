#!/usr/bin/env python3
"""
CIRCT Bug 批量分析脚本

用法:
    python analyze_circt_bug.py /path/to/crash/folder              # 单个崩溃用例
    python analyze_circt_bug.py /path/to/crash/folder1 folder2...  # 多个崩溃用例
    python analyze_circt_bug.py /path/to/crashes/*                 # 通配符
    python analyze_circt_bug.py --list-pending                     # 列出待处理
    python analyze_circt_bug.py --continue                         # 继续处理未完成

选项:
    -j, --jobs N        并行线程数 (默认: 1)
    -f, --force         强制重新分析
    --list-pending      列出所有待处理的崩溃用例
    --continue          继续处理所有未完成的用例

日志输出:
    log/<timestamp>/main.log           # 主进程日志
    log/<timestamp>/circt-bN.log       # 各用例的 opencode 日志
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

# 配置
TEMPLATE_DIR = "circt-b0"
WORK_DIR_PREFIX = "circt-b"
SESSION_PREFIX = "circt-bug-reporter"
SKILL_COMMAND = "report-circt-bug"

# 最终状态（不需要再处理）
FINAL_STATES = {
    "not_a_bug",
    "duplicate", 
    "reproduce_failed",
    "report_ready",
    "submitted"
}

# 错误状态（需要重试）
ERROR_STATES = {
    "error"
}


class Color:
    """终端颜色"""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"


print_lock = Lock()


def setup_log_dir() -> Path:
    """创建带时间戳的日志目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("log") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_main_logger(log_dir: Path) -> logging.Logger:
    """设置主日志记录器"""
    logger = logging.getLogger("circt_bug_main")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_dir / "main.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    
    return logger


def sanitize_dir_name(name: str) -> str:
    """清理目录名，移除不安全字符"""
    safe = re.sub(r'[^\w\-.]', '_', name)
    return safe.strip('_') or 'unnamed'


def get_status(work_dir: Path) -> dict | None:
    """读取工作目录的状态"""
    status_file = work_dir / "status.json"
    if status_file.exists():
        try:
            return json.loads(status_file.read_text())
        except json.JSONDecodeError:
            return None
    return None


def should_skip(work_dir: Path, force: bool) -> tuple[bool, str]:
    """检查是否应该跳过该用例"""
    if not work_dir.exists():
        return False, ""
    
    if force:
        return False, ""
    
    status = get_status(work_dir)
    if status:
        s = status.get("status")
        if s in FINAL_STATES:
            return True, s
    
    return False, ""


def find_crash_files(crash_dir: Path) -> tuple[Path | None, Path | None]:
    """在崩溃目录中查找测例和错误日志"""
    source_file = None
    error_file = None
    
    # 查找测例文件
    for pattern in ["*.sv", "*.fir", "*.mlir", "source.*", "test.*", "bug.*"]:
        files = list(crash_dir.glob(pattern))
        if files:
            source_file = files[0]
            break
    
    # 查找错误日志
    for pattern in ["error.txt", "error.log", "*.log", "stderr.txt", "crash.log"]:
        files = list(crash_dir.glob(pattern))
        if files:
            error_file = files[0]
            break
    
    return source_file, error_file


def init_work_dir(crash_dir: Path, force: bool, logger: logging.Logger) -> tuple[Path, str]:
    """初始化工作目录
    
    Returns:
        (work_dir, bug_id) - bug_id is sanitized crash folder name
    """
    bug_id = sanitize_dir_name(crash_dir.name)
    work_dir = Path(f"{WORK_DIR_PREFIX}{bug_id}")
    
    # 如果目录存在且 force，删除重建
    if work_dir.exists() and force:
        logger.info(f"[{bug_id}] 强制模式，删除旧目录")
        shutil.rmtree(work_dir)
    
    # 复制模板
    if not work_dir.exists():
        if not Path(TEMPLATE_DIR).exists():
            raise FileNotFoundError(f"模板目录 {TEMPLATE_DIR} 不存在")
        
        shutil.copytree(TEMPLATE_DIR, work_dir)
        logger.info(f"[{bug_id}] 从模板创建工作目录: {work_dir}")
    
    # 创建 origin 子目录并复制崩溃文件
    origin_dir = work_dir / "origin"
    origin_dir.mkdir(exist_ok=True)
    
    # 复制整个崩溃目录内容到 origin
    for item in crash_dir.iterdir():
        dest = origin_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    
    logger.info(f"[{bug_id}] 复制崩溃文件到 {origin_dir}")
    
    return work_dir, bug_id


def run_opencode(bug_id: str, work_dir: Path, issue_log: Path, logger: logging.Logger) -> bool:
    """运行 opencode 进行分析"""
    session_name = f"{SESSION_PREFIX}-{bug_id}"
    issue_log_abs = issue_log.resolve()
    work_dir_abs = work_dir.resolve()
        # OPENCODE_CMD="opencode run --title \"$WORK_DIR\" --format json \"/ralph-loop /circt-bug-reporter $CRASH_DIR 。工作文件夹为 $WORK_DIR。 不要提交 issue! do not submit issue to github!\" < /dev/null >> \"$LOG_FILE\" 2>&1"
    cmd = (
        f'cd "{work_dir_abs}" && '
        f'opencode run --title "{session_name}" '
        f'--format json "/ralph-loop /circt-bug-reporter ./origin 。运行完整分析流程直到生成 issue 报告为止。 不要提交 issue! do not submit issue to github!" '
        f'< /dev/null >> "{issue_log_abs}" 2>&1'
    )
    
    logger.info(f"[{bug_id}] 执行 opencode, 日志: {issue_log}")
    logger.debug(f"[{bug_id}] 命令: {cmd}")
    
    try:
        result = subprocess.run(
            ["/bin/zsh", "-lc", cmd],
            stdin=subprocess.DEVNULL,
            timeout=3600,  # 1 小时超时
        )
        logger.info(f"[{bug_id}] opencode 返回码: {result.returncode}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.error(f"[{bug_id}] 超时 (3600s)")
        return False
    except Exception as e:
        logger.error(f"[{bug_id}] 异常: {e}")
        return False


def process_crash(
    crash_dir: Path, 
    force: bool, 
    log_dir: Path, 
    logger: logging.Logger,
    total: int = 0
) -> tuple[str, str]:
    """处理单个崩溃用例
    
    Returns:
        (bug_id, result_string) - bug_id is string (crash folder name)
    """
    crash_dir = Path(crash_dir).resolve()
    
    if not crash_dir.exists():
        logger.error(f"崩溃目录不存在: {crash_dir}")
        return "", "error:not_found"
    
    if not crash_dir.is_dir():
        logger.error(f"不是目录: {crash_dir}")
        return "", "error:not_dir"
    
    try:
        work_dir, bug_id = init_work_dir(crash_dir, force, logger)
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        return "", f"error:init:{e}"
    
    issue_log = log_dir / f"circt-b{bug_id}.log"
    
    if total > 0:
        print_start(bug_id, total, crash_dir.name)
    
    success = run_opencode(bug_id, work_dir, issue_log, logger)
    
    if success:
        status = get_status(work_dir)
        final_status = status.get("status", "unknown") if status else "unknown"
        logger.info(f"[{bug_id}] 完成: {final_status}")
        return bug_id, f"done:{final_status}"
    
    logger.error(f"[{bug_id}] opencode 失败")
    return bug_id, "error:opencode"


def print_start(bug_id: str, total: int, name: str):
    """打印开始处理消息"""
    with print_lock:
        print(f"{Color.CYAN}[?/{total}]{Color.RESET} 正在处理 {name} -> circt-b{bug_id}...", flush=True)


def print_progress(current: int, total: int, bug_id: str, result: str, name: str = ""):
    """打印进度"""
    with print_lock:
        prefix = f"[{current}/{total}]"
        bug_label = f"circt-b{bug_id}" if bug_id else "ERROR"
        name_str = f" ({name})" if name else ""
        
        if result.startswith("skipped"):
            status = result.split(":")[1]
            print(f"{Color.YELLOW}{prefix}{Color.RESET} {bug_label}{name_str} ⏭️  跳过 ({status})", flush=True)
        elif result.startswith("done"):
            status = result.split(":")[1]
            color = Color.GREEN if status == "report_ready" else Color.CYAN
            print(f"{color}{prefix}{Color.RESET} {bug_label}{name_str} ✅ 完成 ({status})", flush=True)
        else:
            print(f"{Color.RED}{prefix}{Color.RESET} {bug_label}{name_str} ❌ 失败 ({result})", flush=True)


def list_pending_work_dirs() -> list[Path]:
    """列出所有待处理/未完成的工作目录"""
    pending = []
    
    for item in sorted(Path(".").iterdir()):
        if not item.is_dir():
            continue
        if not item.name.startswith(WORK_DIR_PREFIX):
            continue
        
        # 跳过模板目录
        if item.name == TEMPLATE_DIR:
            continue
        
        status = get_status(item)
        if status is None:
            pending.append(item)
        elif status.get("status") not in FINAL_STATES:
            pending.append(item)
    
    return pending


def continue_pending(
    force: bool,
    log_dir: Path,
    logger: logging.Logger,
    jobs: int
) -> dict:
    """继续处理所有待处理的工作目录"""
    pending = list_pending_work_dirs()
    
    if not pending:
        print(f"{Color.GREEN}没有待处理的用例{Color.RESET}")
        return {"done": 0, "skipped": 0, "error": 0}
    
    total = len(pending)
    print(f"\n找到 {total} 个待处理用例\n")
    
    stats = {"done": 0, "skipped": 0, "error": 0}
    
    for i, work_dir in enumerate(pending, 1):
        # 提取 bug_id (工作目录名去掉前缀)
        bug_id = work_dir.name[len(WORK_DIR_PREFIX):]
        
        print(f"{Color.CYAN}[{i}/{total}]{Color.RESET} 继续处理 {work_dir.name}...", flush=True)
        
        issue_log = log_dir / f"{work_dir.name}.log"
        success = run_opencode(bug_id, work_dir, issue_log, logger)
        
        if success:
            status = get_status(work_dir)
            final_status = status.get("status", "unknown") if status else "unknown"
            result = f"done:{final_status}"
            stats["done"] += 1
        else:
            result = "error:opencode"
            stats["error"] += 1
        
        print_progress(i, total, bug_id, result, work_dir.name)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="CIRCT Bug 批量分析脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python analyze_circt_bug.py /path/to/crash/folder
    python analyze_circt_bug.py crash1/ crash2/ crash3/
    python analyze_circt_bug.py fuzzer_output/*
    python analyze_circt_bug.py --list-pending
    python analyze_circt_bug.py --continue
        """
    )
    
    parser.add_argument(
        "crash_dirs",
        nargs="*",
        help="崩溃用例文件夹路径"
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=1,
        help="并行线程数 (默认: 1)"
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="强制重新分析"
    )
    parser.add_argument(
        "--list-pending",
        action="store_true",
        help="列出所有待处理的工作目录"
    )
    parser.add_argument(
        "--continue",
        dest="continue_pending",
        action="store_true",
        help="继续处理所有未完成的用例"
    )
    
    args = parser.parse_args()
    
    # 列出待处理
    if args.list_pending:
        pending = list_pending_work_dirs()
        if not pending:
            print(f"{Color.GREEN}没有待处理的用例{Color.RESET}")
        else:
            print(f"\n待处理的用例 ({len(pending)} 个):\n")
            for work_dir in pending:
                status = get_status(work_dir)
                status_str = status.get("status", "no status") if status else "no status.json"
                print(f"  {work_dir.name}: {status_str}")
        return
    
    # 检查模板目录
    if not Path(TEMPLATE_DIR).exists():
        print(f"{Color.RED}错误: 模板目录 {TEMPLATE_DIR} 不存在{Color.RESET}")
        sys.exit(1)
    
    # 设置日志
    log_dir = setup_log_dir()
    logger = setup_main_logger(log_dir)
    
    # 继续处理未完成的
    if args.continue_pending:
        logger.info("继续处理未完成的用例")
        stats = continue_pending(args.force, log_dir, logger, args.jobs)
        
        print(f"\n{'='*50}")
        print(f"完成!")
        print(f"成功: {stats['done']} | 跳过: {stats['skipped']} | 失败: {stats['error']}")
        print(f"日志目录: {log_dir}")
        print(f"{'='*50}\n")
        return
    
    # 检查输入
    if not args.crash_dirs:
        parser.print_help()
        sys.exit(1)
    
    crash_dirs = [Path(d) for d in args.crash_dirs]
    total = len(crash_dirs)
    
    logger.info(f"开始处理 {total} 个崩溃用例, jobs={args.jobs}, force={args.force}")
    logger.info(f"崩溃目录列表: {[str(d) for d in crash_dirs]}")
    
    print(f"\n{'='*50}")
    print(f"CIRCT Bug 分析 - 共 {total} 个")
    print(f"并行数: {args.jobs} | 日志目录: {log_dir}")
    print(f"{'='*50}\n")
    
    stats = {"done": 0, "skipped": 0, "error": 0}
    start_time = datetime.now()
    
    if args.jobs == 1:
        # 串行处理
        for i, crash_dir in enumerate(crash_dirs, 1):
            print(f"{Color.CYAN}[{i}/{total}]{Color.RESET} 正在处理 {crash_dir.name}...", flush=True)
            bug_id, result = process_crash(crash_dir, args.force, log_dir, logger)
            print_progress(i, total, bug_id, result, crash_dir.name)
            
            if result.startswith("done"):
                stats["done"] += 1
            elif result.startswith("skipped"):
                stats["skipped"] += 1
            else:
                stats["error"] += 1
    else:
        # 并行处理
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(process_crash, d, args.force, log_dir, logger, total): d
                for d in crash_dirs
            }
            
            for i, future in enumerate(as_completed(futures), 1):
                crash_dir = futures[future]
                bug_id, result = future.result()
                print_progress(i, total, bug_id, result, crash_dir.name)
                
                if result.startswith("done"):
                    stats["done"] += 1
                elif result.startswith("skipped"):
                    stats["skipped"] += 1
                else:
                    stats["error"] += 1
    
    elapsed = datetime.now() - start_time
    
    logger.info(f"完成: done={stats['done']}, skipped={stats['skipped']}, error={stats['error']}, elapsed={elapsed}")
    
    print(f"\n{'='*50}")
    print(f"完成! 耗时: {elapsed}")
    print(f"成功: {stats['done']} | 跳过: {stats['skipped']} | 失败: {stats['error']}")
    print(f"日志目录: {log_dir}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
