#!/usr/bin/env python3
"""
CIRCT Bug Reproduction Script (Sub-Skill 1)

Usage:
    python3 reproduce.py /path/to/crash/directory

Environment Variables:
    CIRCT_BIN: Path to CIRCT binaries (default: search PATH)
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def find_circt_bin():
    circt_bin = os.environ.get('CIRCT_BIN')
    if circt_bin and Path(circt_bin).exists():
        return Path(circt_bin)
    
    result = shutil.which('circt-verilog')
    if result:
        return Path(result).parent
    
    for p in ['/opt/firtool/bin', '/opt/circt/bin', '/usr/local/bin']:
        if Path(p).joinpath('circt-verilog').exists():
            return Path(p)
    
    return None


def get_next_workdir():
    idx = 1
    while Path(f'./circt-b{idx}').exists():
        idx += 1
    return Path(f'./circt-b{idx}')


def parse_error_txt(error_path: Path) -> dict:
    content = error_path.read_text()
    
    metadata = {}
    
    match = re.search(r'Crash Type:\s*(\w+)', content)
    metadata['crash_type'] = match.group(1) if match else None
    
    match = re.search(r'Hash:\s*([a-f0-9]+)', content)
    metadata['hash'] = match.group(1) if match else None
    
    match = re.search(r'Original Program File:\s*(.+)', content)
    metadata['original_file'] = match.group(1).strip() if match else None
    
    match = re.search(r'--- Compilation Command ---\n(.+?)(?:\n\n|--- Error)', content, re.DOTALL)
    metadata['original_command'] = match.group(1).strip() if match else None
    
    match = re.search(r"Assertion [`'](.+?)[`'] failed", content)
    metadata['assertion_message'] = match.group(1) if match else None
    
    metadata['crashing_tool'] = None
    for tool in ['circt-verilog', 'firtool', 'circt-opt', 'arcilator', 'opt', 'llc']:
        if f'{tool}:' in content or f'Program arguments: ' in content and tool in content:
            metadata['crashing_tool'] = tool
            break
    
    match = re.search(r'Stack dump:(.+?)(?:Aborted|$)', content, re.DOTALL)
    metadata['stack_trace'] = match.group(1).strip() if match else None
    
    return metadata


def extract_reproduction_command(metadata: dict, source_file: Path, circt_bin: Path) -> str:
    original_cmd = metadata.get('original_command', '')
    if not original_cmd:
        return f'{circt_bin}/circt-verilog {source_file}'
    
    parts = original_cmd.split('|')
    crashing_tool = metadata.get('crashing_tool', 'circt-verilog')
    
    crashing_idx = -1
    for i, part in enumerate(parts):
        if crashing_tool in part:
            crashing_idx = i
            break
    
    if crashing_idx >= 0:
        parts = parts[:crashing_idx + 1]
    
    new_parts = []
    for part in parts:
        part = part.strip()
        part = re.sub(r'/[^\s]+/bin/(circt-verilog|firtool|circt-opt|arcilator|opt|llc)', 
                     rf'{circt_bin}/\1', part)
        part = re.sub(r'/tmp/[^\s]+\.sv', str(source_file), part)
        part = re.sub(r'-o\s+/tmp/[^\s]+', '', part)
        new_parts.append(part)
    
    return ' | '.join(new_parts)


def compute_crash_signature(output: str) -> str:
    match = re.search(r"Assertion [`'](.+?)[`'] failed", output)
    assertion = match.group(1) if match else ''
    
    frames = re.findall(r'#\d+\s+0x[a-f0-9]+\s+(\S+)', output)
    key_frames = frames[10:15] if len(frames) > 15 else frames
    
    sig_str = f"{assertion}|{'|'.join(key_frames)}"
    return hashlib.md5(sig_str.encode()).hexdigest()[:12]


def run_reproduction(command: str, workdir: Path, timeout: int = 60) -> tuple:
    log_path = workdir / 'reproduce.log'
    
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=workdir
        )
        output = result.stdout + result.stderr
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        output = "TIMEOUT: Command did not complete within timeout"
        returncode = -1
    except Exception as e:
        output = f"ERROR: {str(e)}"
        returncode = -2
    
    log_path.write_text(output)
    return output, returncode


def main():
    parser = argparse.ArgumentParser(description='Reproduce CIRCT crash')
    parser.add_argument('crash_dir', type=Path, help='Path to crash directory')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout in seconds')
    args = parser.parse_args()
    
    crash_dir = args.crash_dir.resolve()
    error_txt = crash_dir / 'error.txt'
    source_sv = crash_dir / 'source.sv'
    
    if not error_txt.exists():
        print(f"❌ Error: {error_txt} not found")
        sys.exit(1)
    if not source_sv.exists():
        print(f"❌ Error: {source_sv} not found")
        sys.exit(1)
    
    circt_bin = find_circt_bin()
    if not circt_bin:
        print("❌ Error: CIRCT binaries not found. Set CIRCT_BIN environment variable.")
        sys.exit(1)
    print(f"✅ Using CIRCT binaries from: {circt_bin}")
    
    workdir = get_next_workdir().resolve()
    workdir.mkdir(parents=True)
    print(f"✅ Created work directory: {workdir}")
    
    shutil.copy(error_txt, workdir / 'error.txt')
    shutil.copy(source_sv, workdir / 'source.sv')
    print("✅ Copied crash files")
    
    metadata = parse_error_txt(error_txt)
    original_signature = compute_crash_signature(error_txt.read_text())
    
    try:
        version_result = subprocess.run(
            [str(circt_bin / 'circt-verilog'), '--version'],
            capture_output=True, text=True
        )
        circt_version = version_result.stdout.strip() or version_result.stderr.strip()
    except:
        circt_version = "unknown"
    
    repro_cmd = extract_reproduction_command(metadata, (workdir / 'source.sv').resolve(), circt_bin)
    print(f"✅ Reproduction command: {repro_cmd}")
    
    print("🔄 Running reproduction...")
    output, returncode = run_reproduction(repro_cmd, workdir, args.timeout)
    
    repro_signature = compute_crash_signature(output)
    reproduced = returncode != 0 and ('Assertion' in output or 'PLEASE submit a bug report' in output)
    same_crash = original_signature == repro_signature
    
    metadata['reproduction'] = {
        'command': repro_cmd,
        'circt_bin': str(circt_bin),
        'circt_version': circt_version,
        'timestamp': datetime.now().isoformat(),
        'returncode': returncode,
        'reproduced': reproduced,
        'same_crash': same_crash,
        'original_signature': original_signature,
        'repro_signature': repro_signature,
    }
    metadata['workdir'] = str(workdir)
    metadata['source_dir'] = str(crash_dir)
    
    (workdir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    
    if reproduced:
        if same_crash:
            print(f"✅ Bug REPRODUCED (same crash signature)")
        else:
            print(f"⚠️  Bug reproduced but crash signature differs")
            print(f"   Original: {original_signature}")
            print(f"   Current:  {repro_signature}")
    else:
        print(f"❌ Bug NOT reproduced (returncode: {returncode})")
        print(f"   Check {workdir}/reproduce.log for details")
    
    print(f"\n📁 Work directory: {workdir}")
    print(f"   Next step: python3 minimize.py {workdir}")
    
    return 0 if reproduced else 1


if __name__ == '__main__':
    sys.exit(main())
