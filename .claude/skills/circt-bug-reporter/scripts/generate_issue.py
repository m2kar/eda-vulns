#!/usr/bin/env python3
"""
CIRCT Issue Report Generator (Sub-Skill 6)

Generates issue.md including root cause analysis results.

Usage:
    python3 generate_issue.py ./circt-b<id>
"""

import argparse
import json
import re
import sys
from pathlib import Path


DIALECT_LABELS = {
    'MooreToCore': 'Moore',
    'moore::': 'Moore',
    'firrtl::': 'FIRRTL',
    'arc::': 'Arc',
    'hw::': 'HW',
    'seq::': 'Seq',
    'sv::': 'SV',
    'comb::': 'comb',
    'llhd::': 'LLHD',
    'calyx::': 'Calyx',
    'handshake::': 'Handshake',
}


def detect_dialect(error_text: str, stack_trace: str = '') -> str:
    combined = error_text + (stack_trace or '')
    
    for pattern, label in DIALECT_LABELS.items():
        if pattern in combined:
            return label
    
    return ''


def extract_key_error(error_text: str) -> str:
    match = re.search(r"Assertion [`'](.+?)[`'] failed", error_text)
    if match:
        return f"Assertion `{match.group(1)}` failed"
    
    match = re.search(r'error:\s*(.+?)(?:\n|$)', error_text)
    if match:
        return match.group(1).strip()
    
    return "Internal compiler error"


def load_error_content(workdir: Path) -> tuple[str, str]:
    error_log = workdir / 'error.log'
    if error_log.exists():
        content = error_log.read_text()
    else:
        error_txt = workdir / 'error.txt'
        if error_txt.exists():
            content = error_txt.read_text()
        else:
            return '', ''
    
    match = re.search(r'Stack dump:(.*?)(?:Aborted|$)', content, re.DOTALL)
    stack_trace = match.group(1).strip() if match else ''
    
    error_lines = []
    for line in content.split('\n'):
        if 'Stack dump:' in line:
            break
        error_lines.append(line)
    error_output = '\n'.join(error_lines[-30:])
    
    return error_output, stack_trace


def load_analysis(workdir: Path) -> dict:
    analysis_path = workdir / 'analysis.json'
    if analysis_path.exists():
        return json.loads(analysis_path.read_text())
    return {}


def load_validation(workdir: Path) -> dict:
    validation_path = workdir / 'validation.json'
    if validation_path.exists():
        return json.loads(validation_path.read_text())
    return {}


def load_duplicates(workdir: Path) -> dict:
    duplicates_path = workdir / 'duplicates.json'
    if duplicates_path.exists():
        return json.loads(duplicates_path.read_text())
    return {}


def generate_issue_title(metadata: dict, analysis: dict, error_text: str) -> str:
    dialect = analysis.get('dialect') or detect_dialect(error_text, '')
    key_error = extract_key_error(error_text)
    
    if dialect:
        return f"[{dialect}] {key_error}"
    
    tool = metadata.get('crashing_tool', 'circt')
    return f"[{tool}] {key_error}"


def generate_description(analysis: dict, error_text: str) -> str:
    key_error = extract_key_error(error_text)
    
    hypotheses = analysis.get('hypotheses', [])
    if hypotheses:
        primary = hypotheses[0]
        return f"{key_error}\n\n**Likely cause**: {primary.get('hypothesis', 'Unknown')}"
    
    return key_error


def generate_root_cause_section(analysis: dict) -> str:
    if not analysis:
        return ""
    
    sections = []
    sections.append("## Root Cause Analysis\n")
    
    dialect = analysis.get('dialect')
    failing_pass = analysis.get('failing_pass')
    crash_pattern = analysis.get('crash_pattern', {})
    
    if dialect:
        sections.append(f"- **Dialect**: {dialect}")
    if failing_pass:
        sections.append(f"- **Failing Pass**: {failing_pass}")
    if crash_pattern.get('category'):
        sections.append(f"- **Crash Category**: {crash_pattern['category']}")
        if crash_pattern.get('likely_cause'):
            sections.append(f"- **Likely Cause**: {crash_pattern['likely_cause']}")
    
    hypotheses = analysis.get('hypotheses', [])
    if hypotheses:
        sections.append("\n### Hypotheses\n")
        for i, h in enumerate(hypotheses[:3], 1):
            confidence = h.get('confidence', 'unknown')
            sections.append(f"{i}. **{h.get('hypothesis', 'Unknown')}** (confidence: {confidence})")
            if h.get('evidence'):
                sections.append(f"   - Evidence: {h['evidence']}")
    
    test_case = analysis.get('test_case_analysis', {})
    if test_case.get('key_features'):
        sections.append(f"\n### Key Features in Test Case\n")
        for feature in test_case['key_features']:
            sections.append(f"- {feature}")
    
    return '\n'.join(sections)


def generate_validation_note(validation: dict) -> str:
    if not validation:
        return ""
    
    classification = validation.get('classification', {})
    
    if classification.get('is_unsupported_feature'):
        return "\n> **Note**: This test case uses features that may not yet be fully supported.\n"
    
    other_tools = validation.get('other_tools', {})
    tools_accept = [t for t, r in other_tools.items() if r and r.get('success')]
    
    if tools_accept:
        return f"\n> **Validation**: Test case accepted by: {', '.join(tools_accept)}\n"
    
    return ""


def generate_duplicates_note(duplicates: dict) -> str:
    if not duplicates:
        return ""
    
    high_sim = duplicates.get('high_similarity', [])
    
    if high_sim:
        notes = ["\n> **Related Issues**:"]
        for issue in high_sim[:3]:
            notes.append(f"> - #{issue['number']}: {issue['title']}")
        return '\n'.join(notes) + '\n'
    
    return ""


def generate_issue_body(
    metadata: dict, 
    analysis: dict,
    validation: dict,
    duplicates: dict,
    test_code: str, 
    command: str, 
    error_output: str, 
    stack_trace: str, 
    circt_version: str
) -> str:
    description = generate_description(analysis, error_output)
    
    file_ext = metadata.get('minimization', {}).get('bug_file', 'test.sv')
    if file_ext.endswith('.fir'):
        lang = 'firrtl'
        save_as = 'test.fir'
    elif file_ext.endswith('.mlir'):
        lang = 'mlir'
        save_as = 'test.mlir'
    else:
        lang = 'systemverilog'
        save_as = 'test.sv'
    
    body_parts = []
    
    body_parts.append(f"""## Description

{description}
""")
    
    validation_note = generate_validation_note(validation)
    if validation_note:
        body_parts.append(validation_note)
    
    duplicates_note = generate_duplicates_note(duplicates)
    if duplicates_note:
        body_parts.append(duplicates_note)
    
    body_parts.append(f"""## Steps to Reproduce

1. Save the following code as `{save_as}`
2. Run: `{command}`

## Test Case

```{lang}
{test_code}
```

## Error Output

```
{error_output}
```
""")
    
    root_cause = generate_root_cause_section(analysis)
    if root_cause:
        body_parts.append(root_cause)
    
    body_parts.append(f"""
## Environment

- **CIRCT Version**: {circt_version}
""")
    
    if stack_trace:
        body_parts.append(f"""
<details>
<summary>Stack Trace</summary>

```
{stack_trace}
```

</details>
""")
    
    return '\n'.join(body_parts)


def main():
    parser = argparse.ArgumentParser(description='Generate CIRCT issue report')
    parser.add_argument('workdir', type=Path, help='Work directory')
    args = parser.parse_args()
    
    workdir = args.workdir.resolve()
    
    metadata_path = workdir / 'metadata.json'
    if not metadata_path.exists():
        print(f"❌ Error: {metadata_path} not found")
        sys.exit(1)
    
    metadata = json.loads(metadata_path.read_text())
    
    test_file = None
    for name in ['bug.sv', 'bug.fir', 'bug.mlir', 'minimal.sv', 'source.sv']:
        candidate = workdir / name
        if candidate.exists():
            test_file = candidate
            break
    
    if not test_file:
        print(f"❌ Error: No test case found")
        sys.exit(1)
    
    test_code = test_file.read_text()
    
    command_txt = workdir / 'command.txt'
    if command_txt.exists():
        command = command_txt.read_text().strip()
    else:
        command = metadata.get('reproduction', {}).get('command', 'circt-verilog test.sv')
    
    error_output, stack_trace = load_error_content(workdir)
    
    analysis = load_analysis(workdir)
    validation = load_validation(workdir)
    duplicates = load_duplicates(workdir)
    
    circt_version = metadata.get('reproduction', {}).get('circt_version', 'unknown')
    
    dialect = analysis.get('dialect') or detect_dialect(error_output, stack_trace)
    
    title = generate_issue_title(metadata, analysis, error_output)
    body = generate_issue_body(
        metadata,
        analysis,
        validation,
        duplicates,
        test_code,
        command,
        error_output,
        stack_trace,
        circt_version
    )
    
    labels = ['bug']
    if dialect:
        labels.append(dialect)
    
    issue_md = workdir / 'issue.md'
    issue_content = f"""# {title}

{body}

---
**Labels**: {', '.join(labels)}
"""
    issue_md.write_text(issue_content)
    
    metadata['issue'] = {
        'title': title,
        'labels': labels,
        'dialect': dialect,
        'includes_root_cause': bool(analysis),
        'includes_validation': bool(validation),
        'includes_duplicates_check': bool(duplicates),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    
    print(f"✅ Generated issue report: {issue_md}")
    print(f"   Title: {title}")
    print(f"   Labels: {', '.join(labels)}")
    
    if analysis:
        print(f"   Root Cause Analysis: ✅ Included")
    if validation:
        print(f"   Validation: ✅ Included")
    if duplicates:
        print(f"   Duplicate Check: ✅ Included")
    
    print(f"\n📋 Please review {issue_md} before submitting")
    print(f"   Next step: python3 submit_issue.py {workdir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
