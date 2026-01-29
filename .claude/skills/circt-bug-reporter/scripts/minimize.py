#!/usr/bin/env python3
"""
CIRCT Test Case Minimization Script (Sub-Skill 3)

Enhanced version that:
- Uses root cause analysis to guide minimization
- Removes all unrelated code while preserving bug reproduction
- Outputs bug.sv (or bug.fir/.mlir) + error.log as final artifacts
- Deletes original source.sv and error.txt after successful minimization

Usage:
    python3 minimize.py ./circt-b<id>
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def find_circt_bin(metadata: dict):
    circt_bin = metadata.get('reproduction', {}).get('circt_bin')
    if circt_bin and Path(circt_bin).exists():
        return Path(circt_bin)
    
    circt_bin = os.environ.get('CIRCT_BIN')
    if circt_bin and Path(circt_bin).exists():
        return Path(circt_bin)
    
    result = shutil.which('circt-verilog')
    if result:
        return Path(result).parent
    
    return None


def verify_crash(command: str, workdir: Path, expected_assertion: str = '') -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=30, cwd=workdir
        )
        output = result.stdout + result.stderr
        
        if 'Assertion' not in output and 'PLEASE submit a bug report' not in output:
            return False, output
        
        if expected_assertion and expected_assertion not in output:
            return False, output
        
        return True, output
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


def get_key_constructs_from_analysis(workdir: Path) -> list:
    analysis_path = workdir / 'analysis.json'
    if not analysis_path.exists():
        return []
    
    try:
        analysis = json.loads(analysis_path.read_text())
        test_case = analysis.get('test_case_analysis', {})
        
        constructs = []
        constructs.extend(test_case.get('key_features', []))
        constructs.extend(test_case.get('constructs', []))
        
        return constructs
    except:
        return []


def get_keywords_from_analysis(workdir: Path) -> list:
    analysis_path = workdir / 'analysis.json'
    if not analysis_path.exists():
        return []
    
    try:
        analysis = json.loads(analysis_path.read_text())
        return analysis.get('keywords', [])
    except:
        return []


def is_essential_line(line: str, key_constructs: list, keywords: list) -> bool:
    line_lower = line.lower().strip()
    
    if not line_lower:
        return False
    
    essential_patterns = [
        r'^\s*module\s+\w+',
        r'^\s*endmodule',
        r'^\s*input\s+',
        r'^\s*output\s+',
        r'^\s*typedef\s+',
    ]
    
    for pattern in essential_patterns:
        if re.match(pattern, line, re.IGNORECASE):
            return True
    
    for construct in key_constructs:
        construct_keywords = construct.lower().split()
        if any(kw in line_lower for kw in construct_keywords):
            return True
    
    for keyword in keywords:
        if keyword.lower() in line_lower:
            return True
    
    return False


def minimize_sv_code_smart(source_path: Path, command_template: str, workdir: Path, 
                           expected_assertion: str, key_constructs: list, keywords: list) -> str:
    original = source_path.read_text()
    current = original
    
    test_file = workdir / 'test_minimize.sv'
    
    def test_variant(code: str) -> bool:
        test_file.write_text(code)
        cmd = command_template.replace(str(source_path), str(test_file))
        success, _ = verify_crash(cmd, workdir, expected_assertion)
        return success
    
    lines = current.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if is_essential_line(line, key_constructs, keywords):
            i += 1
            continue
        
        if not line or line.startswith('//'):
            candidate = '\n'.join(lines[:i] + lines[i+1:])
            if candidate.strip() and test_variant(candidate):
                lines = lines[:i] + lines[i+1:]
                continue
        i += 1
    
    current = '\n'.join(lines)
    
    comment_patterns = [
        r'\s*//[^\n]*',
        r'/\*[\s\S]*?\*/',
    ]
    
    for pattern in comment_patterns:
        candidate = re.sub(pattern, '', current, flags=re.MULTILINE)
        if candidate.strip() and test_variant(candidate):
            current = candidate
    
    lines = current.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('wire ') or line.startswith('reg ') or line.startswith('logic '):
            var_match = re.match(r'(?:wire|reg|logic)\s+(?:\[\d+:\d+\])?\s*(\w+)', line)
            if var_match:
                var_name = var_match.group(1)
                rest_of_code = '\n'.join(lines[:i] + lines[i+1:])
                if var_name not in rest_of_code:
                    candidate = rest_of_code
                    if candidate.strip() and test_variant(candidate):
                        lines = lines[:i] + lines[i+1:]
                        continue
        i += 1
    
    current = '\n'.join(lines)
    
    candidate = re.sub(r'\n\s*\n\s*\n+', '\n\n', current)
    if candidate.strip() and test_variant(candidate):
        current = candidate
    
    candidate = re.sub(r'\s+$', '', current, flags=re.MULTILINE)
    if candidate.strip() and test_variant(candidate):
        current = candidate
    
    test_file.unlink(missing_ok=True)
    
    return current.strip()


def simplify_command(original_command: str, source_file: Path, circt_bin: Path) -> str:
    parts = original_command.split('|')
    
    crash_idx = len(parts)
    for i, part in enumerate(parts):
        if 'circt-verilog' in part or 'firtool' in part or 'circt-opt' in part:
            cmd = ' | '.join(parts[:i+1])
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=30
                )
                if 'Assertion' in result.stderr or 'PLEASE submit' in result.stderr:
                    crash_idx = i + 1
                    break
            except:
                continue
    
    parts = parts[:crash_idx]
    
    simplified_parts = []
    for part in parts:
        part = part.strip()
        part = re.sub(r'/[^\s]+/bin/(circt-verilog|firtool|circt-opt|arcilator)', 
                     r'\1', part)
        part = re.sub(r'/[^\s]+\.sv', str(source_file.name), part)
        part = re.sub(r'-o\s+[^\s]+', '', part)
        part = re.sub(r'\s+', ' ', part).strip()
        simplified_parts.append(part)
    
    return ' | '.join(simplified_parts)


def extract_minimal_error_log(reproduce_output: str) -> str:
    lines = reproduce_output.split('\n')
    
    error_start = -1
    for i, line in enumerate(lines):
        if 'error:' in line.lower() or 'assertion' in line.lower() or 'PLEASE submit' in line:
            error_start = i
            break
    
    if error_start == -1:
        return reproduce_output
    
    error_lines = []
    stack_started = False
    
    for line in lines[error_start:]:
        error_lines.append(line)
        
        if 'Stack dump:' in line:
            stack_started = True
        
        if stack_started and ('Aborted' in line or line.strip() == ''):
            break
    
    return '\n'.join(error_lines)


def main():
    parser = argparse.ArgumentParser(description='Minimize CIRCT crash test case')
    parser.add_argument('workdir', type=Path, help='Work directory from reproduce.py')
    parser.add_argument('--keep-originals', action='store_true', 
                        help='Do not delete original source.sv and error.txt')
    args = parser.parse_args()
    
    workdir = args.workdir.resolve()
    
    metadata_path = workdir / 'metadata.json'
    if not metadata_path.exists():
        print(f"❌ Error: {metadata_path} not found. Run reproduce.py first.")
        sys.exit(1)
    
    metadata = json.loads(metadata_path.read_text())
    
    if not metadata.get('reproduction', {}).get('reproduced', False):
        print("❌ Error: Bug was not reproduced. Cannot minimize.")
        sys.exit(1)
    
    circt_bin = find_circt_bin(metadata)
    if not circt_bin:
        print("❌ Error: CIRCT binaries not found.")
        sys.exit(1)
    
    source_sv = workdir / 'source.sv'
    if not source_sv.exists():
        for ext in ['.fir', '.mlir']:
            alt_source = workdir / f'source{ext}'
            if alt_source.exists():
                source_sv = alt_source
                break
    
    if not source_sv.exists():
        print(f"❌ Error: No source file found in {workdir}")
        sys.exit(1)
    
    repro_command = metadata['reproduction']['command']
    expected_assertion = metadata.get('assertion_message', '')
    
    key_constructs = get_key_constructs_from_analysis(workdir)
    keywords = get_keywords_from_analysis(workdir)
    
    if key_constructs or keywords:
        print(f"🧠 Using root cause analysis to guide minimization")
        print(f"   Key constructs: {', '.join(key_constructs[:3])}" if key_constructs else "")
        print(f"   Keywords: {', '.join(keywords[:5])}" if keywords else "")
    
    print("🔄 Minimizing test case...")
    minimized_code = minimize_sv_code_smart(
        source_sv, repro_command, workdir, expected_assertion,
        key_constructs, keywords
    )
    
    suffix = source_sv.suffix
    bug_file = workdir / f'bug{suffix}'
    bug_file.write_text(minimized_code)
    
    original_lines = len(source_sv.read_text().split('\n'))
    minimal_lines = len(minimized_code.split('\n'))
    print(f"✅ Minimized: {original_lines} lines → {minimal_lines} lines")
    
    print("🔄 Simplifying command...")
    simplified_cmd = simplify_command(repro_command, bug_file, circt_bin)
    
    command_txt = workdir / 'command.txt'
    command_txt.write_text(simplified_cmd)
    print(f"✅ Simplified command: {simplified_cmd}")
    
    print("🔄 Verifying minimized test case...")
    test_cmd = repro_command.replace(str(source_sv), str(bug_file))
    verified, reproduce_output = verify_crash(test_cmd, workdir, expected_assertion)
    
    if verified:
        print("✅ Minimized test case verified")
        
        error_log = workdir / 'error.log'
        minimal_error = extract_minimal_error_log(reproduce_output)
        error_log.write_text(minimal_error)
        print(f"✅ Error log: {error_log}")
        
        if not args.keep_originals:
            original_error = workdir / 'error.txt'
            if original_error.exists():
                original_error.unlink()
                print(f"🗑️  Deleted original: error.txt")
            
            if source_sv.name != bug_file.name:
                source_sv.unlink()
                print(f"🗑️  Deleted original: {source_sv.name}")
            
            minimal_sv = workdir / 'minimal.sv'
            if minimal_sv.exists() and minimal_sv != bug_file:
                minimal_sv.unlink()
    else:
        print("⚠️  Warning: Minimized test case may not trigger the same crash")
        print("   Keeping original files and using original as bug file")
        shutil.copy(source_sv, bug_file)
        
        error_txt = workdir / 'error.txt'
        if error_txt.exists():
            shutil.copy(error_txt, workdir / 'error.log')
    
    metadata['minimization'] = {
        'original_lines': original_lines,
        'minimal_lines': minimal_lines,
        'reduction_ratio': f"{(1 - minimal_lines/original_lines)*100:.1f}%",
        'simplified_command': simplified_cmd,
        'bug_file': bug_file.name,
        'error_log': 'error.log',
        'verified': verified,
        'used_root_cause_analysis': bool(key_constructs or keywords),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    
    print(f"\n📁 Output files:")
    print(f"   {bug_file}")
    print(f"   {workdir / 'error.log'}")
    print(f"   {command_txt}")
    print(f"\n   Next step: python3 validate_testcase.py {workdir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
