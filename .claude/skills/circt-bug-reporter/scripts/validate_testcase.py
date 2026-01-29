#!/usr/bin/env python3
"""
CIRCT Test Case Validation Script (Sub-Skill 4)

Validates whether the test case is syntactically correct according to IEEE 1800-2005
and determines if the bug is:
1. A genuine CIRCT bug (test case is valid but CIRCT crashes)
2. A special/intentional CIRCT design limitation
3. An invalid test case (syntax error or unsupported construct)

Uses:
- Root cause analysis results
- IEEE 1800-2005 SystemVerilog standard reference
- CIRCT official documentation on supported features

Usage:
    python3 validate_testcase.py ./circt-b<id>
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


KNOWN_UNSUPPORTED_FEATURES = {
    'dynamic_arrays': {
        'pattern': r'\[\s*\$\s*\]',
        'description': 'Dynamic arrays (unsized)',
        'circt_status': 'Not yet supported in Moore dialect',
        'ieee_section': '7.5.1',
    },
    'associative_arrays': {
        'pattern': r'\[\s*\*\s*\]|\[\s*string\s*\]',
        'description': 'Associative arrays',
        'circt_status': 'Not yet supported in Moore dialect',
        'ieee_section': '7.8',
    },
    'classes': {
        'pattern': r'\bclass\s+\w+',
        'description': 'SystemVerilog classes',
        'circt_status': 'Not supported (not synthesizable)',
        'ieee_section': '8.0',
    },
    'covergroups': {
        'pattern': r'\bcovergroup\b',
        'description': 'Coverage constructs',
        'circt_status': 'Not supported (verification only)',
        'ieee_section': '19.0',
    },
    'assertions_concurrent': {
        'pattern': r'\bassert\s+property\b',
        'description': 'Concurrent assertions',
        'circt_status': 'Partial support',
        'ieee_section': '16.0',
    },
    'program_blocks': {
        'pattern': r'\bprogram\s+\w+',
        'description': 'Program blocks',
        'circt_status': 'Not supported (verification only)',
        'ieee_section': '24.0',
    },
    'randomize': {
        'pattern': r'\brandomize\b|\brand\b\s+\w+',
        'description': 'Randomization',
        'circt_status': 'Not supported (verification only)',
        'ieee_section': '18.0',
    },
    'dpi': {
        'pattern': r'\bimport\s+"DPI',
        'description': 'DPI (Direct Programming Interface)',
        'circt_status': 'Not yet supported',
        'ieee_section': '35.0',
    },
}

SYNTAX_ISSUES = {
    'missing_semicolon': {
        'pattern': r'(?:end|endmodule|endfunction|endtask)\s*[^\s;]',
        'description': 'Missing semicolon after keyword',
        'valid_sv': False,
    },
    'unmatched_begin_end': {
        'description': 'Unmatched begin/end blocks',
        'valid_sv': False,
    },
    'invalid_port_direction': {
        'pattern': r'\b(input|output|inout)\s+(?!logic|reg|wire|integer|real|\[)',
        'description': 'Invalid port declaration',
        'valid_sv': False,
    },
}

DIALECT_LIMITATIONS = {
    'Moore': {
        'union_nested': {
            'description': 'Nested unions in structs',
            'status': 'known_limitation',
            'github_issue': None,
        },
        'array_sensitivity': {
            'description': 'Array element in sensitivity list',
            'status': 'known_limitation',
            'github_issue': 'https://github.com/llvm/circt/issues/9469',
        },
    },
    'FIRRTL': {
        'async_reset_init': {
            'description': 'Async reset with initial value',
            'status': 'design_decision',
        },
    },
}


def check_other_tools(test_file: Path) -> dict:
    results = {
        'verilator': None,
        'iverilog': None,
        'slang': None,
    }
    
    if shutil.which('verilator'):
        try:
            result = subprocess.run(
                ['verilator', '--lint-only', '-Wall', str(test_file)],
                capture_output=True, text=True, timeout=30
            )
            results['verilator'] = {
                'success': result.returncode == 0,
                'errors': result.stderr if result.returncode != 0 else None,
            }
        except (subprocess.TimeoutExpired, Exception) as e:
            results['verilator'] = {'success': None, 'error': str(e)}
    
    if shutil.which('iverilog'):
        try:
            result = subprocess.run(
                ['iverilog', '-g2012', '-t', 'null', str(test_file)],
                capture_output=True, text=True, timeout=30
            )
            results['iverilog'] = {
                'success': result.returncode == 0,
                'errors': result.stderr if result.returncode != 0 else None,
            }
        except (subprocess.TimeoutExpired, Exception) as e:
            results['iverilog'] = {'success': None, 'error': str(e)}
    
    if shutil.which('slang'):
        try:
            result = subprocess.run(
                ['slang', '--syntax-only', str(test_file)],
                capture_output=True, text=True, timeout=30
            )
            results['slang'] = {
                'success': result.returncode == 0,
                'errors': result.stderr if result.returncode != 0 else None,
            }
        except (subprocess.TimeoutExpired, Exception) as e:
            results['slang'] = {'success': None, 'error': str(e)}
    
    return results


def check_unsupported_features(code: str) -> list:
    found = []
    
    for feature_id, feature in KNOWN_UNSUPPORTED_FEATURES.items():
        if re.search(feature['pattern'], code):
            found.append({
                'feature': feature_id,
                'description': feature['description'],
                'circt_status': feature['circt_status'],
                'ieee_section': feature.get('ieee_section'),
            })
    
    return found


def check_syntax_issues(code: str) -> list:
    found = []
    
    for issue_id, issue in SYNTAX_ISSUES.items():
        if 'pattern' in issue:
            if re.search(issue['pattern'], code):
                found.append({
                    'issue': issue_id,
                    'description': issue['description'],
                    'valid_sv': issue['valid_sv'],
                })
    
    begin_count = len(re.findall(r'\bbegin\b', code))
    end_count = len(re.findall(r'\bend\b(?!\w)', code))
    if begin_count != end_count:
        found.append({
            'issue': 'unmatched_begin_end',
            'description': f'Unmatched begin/end ({begin_count} begin vs {end_count} end)',
            'valid_sv': False,
        })
    
    return found


def check_dialect_limitations(code: str, dialect: str, analysis: dict) -> list:
    found = []
    
    limitations = DIALECT_LIMITATIONS.get(dialect, {})
    test_case_analysis = analysis.get('test_case_analysis', {})
    key_features = test_case_analysis.get('key_features', [])
    
    if dialect == 'Moore':
        if 'array in sensitivity list' in key_features:
            lim = limitations.get('array_sensitivity', {})
            found.append({
                'limitation': 'array_sensitivity',
                'description': lim.get('description', 'Array element in sensitivity list'),
                'status': lim.get('status', 'known_limitation'),
                'github_issue': lim.get('github_issue'),
            })
        
        if re.search(r'union\s+packed.*struct|struct\s+packed.*union', code, re.DOTALL):
            lim = limitations.get('union_nested', {})
            found.append({
                'limitation': 'union_nested',
                'description': lim.get('description', 'Nested unions in structs'),
                'status': lim.get('status', 'known_limitation'),
            })
    
    return found


def determine_bug_classification(validation: dict) -> dict:
    classification = {
        'is_valid_testcase': True,
        'is_genuine_bug': True,
        'is_design_limitation': False,
        'is_unsupported_feature': False,
        'recommendation': 'report',
        'confidence': 'high',
        'reasoning': [],
    }
    
    if validation.get('syntax_issues'):
        for issue in validation['syntax_issues']:
            if not issue.get('valid_sv', True):
                classification['is_valid_testcase'] = False
                classification['is_genuine_bug'] = False
                classification['recommendation'] = 'fix_testcase'
                classification['reasoning'].append(
                    f"Syntax issue: {issue['description']}"
                )
    
    if validation.get('unsupported_features'):
        classification['is_unsupported_feature'] = True
        for feature in validation['unsupported_features']:
            classification['reasoning'].append(
                f"Uses unsupported feature: {feature['description']} ({feature['circt_status']})"
            )
        
        if not validation.get('dialect_limitations'):
            classification['recommendation'] = 'report_as_feature_request'
            classification['confidence'] = 'medium'
    
    if validation.get('dialect_limitations'):
        for lim in validation['dialect_limitations']:
            if lim.get('status') == 'design_decision':
                classification['is_design_limitation'] = True
                classification['is_genuine_bug'] = False
                classification['recommendation'] = 'not_a_bug'
                classification['reasoning'].append(
                    f"Known design limitation: {lim['description']}"
                )
            elif lim.get('github_issue'):
                classification['recommendation'] = 'check_existing_issue'
                classification['reasoning'].append(
                    f"Matches known issue: {lim['github_issue']}"
                )
    
    other_tools = validation.get('other_tools', {})
    tools_that_accept = []
    tools_that_reject = []
    
    for tool, result in other_tools.items():
        if result and result.get('success') is not None:
            if result['success']:
                tools_that_accept.append(tool)
            else:
                tools_that_reject.append(tool)
    
    if tools_that_accept and not tools_that_reject:
        classification['reasoning'].append(
            f"Other tools ({', '.join(tools_that_accept)}) accept this code"
        )
        classification['is_valid_testcase'] = True
    elif tools_that_reject and not tools_that_accept:
        classification['reasoning'].append(
            f"Other tools ({', '.join(tools_that_reject)}) also reject this code"
        )
        classification['confidence'] = 'medium'
    
    if not classification['reasoning']:
        classification['reasoning'].append(
            "Test case appears valid and triggers a CIRCT crash"
        )
    
    return classification


def generate_validation_report(validation: dict) -> str:
    report = []
    report.append("# Test Case Validation Report\n")
    
    classification = validation.get('classification', {})
    report.append("## Classification\n")
    report.append(f"- **Valid Test Case**: {'✅ Yes' if classification.get('is_valid_testcase') else '❌ No'}")
    report.append(f"- **Genuine Bug**: {'✅ Yes' if classification.get('is_genuine_bug') else '❌ No'}")
    report.append(f"- **Design Limitation**: {'⚠️ Yes' if classification.get('is_design_limitation') else '✅ No'}")
    report.append(f"- **Unsupported Feature**: {'⚠️ Yes' if classification.get('is_unsupported_feature') else '✅ No'}")
    report.append(f"- **Recommendation**: `{classification.get('recommendation', 'unknown')}`")
    report.append(f"- **Confidence**: {classification.get('confidence', 'unknown')}")
    report.append("")
    
    reasoning = classification.get('reasoning', [])
    if reasoning:
        report.append("### Reasoning\n")
        for reason in reasoning:
            report.append(f"- {reason}")
        report.append("")
    
    syntax_issues = validation.get('syntax_issues', [])
    if syntax_issues:
        report.append("## Syntax Issues\n")
        for issue in syntax_issues:
            report.append(f"- **{issue['issue']}**: {issue['description']}")
        report.append("")
    
    unsupported = validation.get('unsupported_features', [])
    if unsupported:
        report.append("## Unsupported Features\n")
        for feature in unsupported:
            report.append(f"### {feature['feature']}")
            report.append(f"- Description: {feature['description']}")
            report.append(f"- CIRCT Status: {feature['circt_status']}")
            if feature.get('ieee_section'):
                report.append(f"- IEEE 1800-2005 Section: {feature['ieee_section']}")
            report.append("")
    
    limitations = validation.get('dialect_limitations', [])
    if limitations:
        report.append("## Dialect Limitations\n")
        for lim in limitations:
            report.append(f"- **{lim['limitation']}**: {lim['description']}")
            if lim.get('github_issue'):
                report.append(f"  - Related issue: {lim['github_issue']}")
        report.append("")
    
    other_tools = validation.get('other_tools', {})
    if any(v for v in other_tools.values()):
        report.append("## Cross-Tool Validation\n")
        for tool, result in other_tools.items():
            if result:
                status = "✅ Accepts" if result.get('success') else "❌ Rejects"
                report.append(f"- **{tool}**: {status}")
                if result.get('errors'):
                    report.append(f"  ```")
                    report.append(f"  {result['errors'][:500]}")
                    report.append(f"  ```")
        report.append("")
    
    return '\n'.join(report)


def main():
    parser = argparse.ArgumentParser(description='Validate CIRCT test case')
    parser.add_argument('workdir', type=Path, help='Work directory')
    parser.add_argument('--skip-tools', action='store_true', 
                        help='Skip cross-validation with other tools')
    args = parser.parse_args()
    
    workdir = args.workdir.resolve()
    
    bug_file = workdir / 'bug.sv'
    if not bug_file.exists():
        for ext in ['.fir', '.mlir']:
            alt = workdir / f'bug{ext}'
            if alt.exists():
                bug_file = alt
                break
        
        if not bug_file.exists():
            bug_file = workdir / 'minimal.sv'
        if not bug_file.exists():
            bug_file = workdir / 'source.sv'
    
    if not bug_file.exists():
        print(f"❌ Error: No test case found in {workdir}")
        sys.exit(1)
    
    code = bug_file.read_text()
    
    analysis_path = workdir / 'analysis.json'
    analysis = {}
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
    
    dialect = analysis.get('dialect', 'Moore')
    
    print(f"🔍 Validating test case: {bug_file.name}")
    print(f"   Dialect: {dialect}")
    
    validation = {
        'test_file': bug_file.name,
        'dialect': dialect,
    }
    
    print("🔄 Checking for unsupported features...")
    validation['unsupported_features'] = check_unsupported_features(code)
    
    print("🔄 Checking syntax...")
    validation['syntax_issues'] = check_syntax_issues(code)
    
    print("🔄 Checking dialect limitations...")
    validation['dialect_limitations'] = check_dialect_limitations(code, dialect, analysis)
    
    if not args.skip_tools:
        print("🔄 Cross-validating with other tools...")
        validation['other_tools'] = check_other_tools(bug_file)
    else:
        validation['other_tools'] = {}
    
    print("🔄 Classifying bug...")
    validation['classification'] = determine_bug_classification(validation)
    
    validation_json = workdir / 'validation.json'
    validation_json.write_text(json.dumps(validation, indent=2))
    
    report = generate_validation_report(validation)
    validation_md = workdir / 'validation.md'
    validation_md.write_text(report)
    
    metadata_path = workdir / 'metadata.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata['validation'] = {
            'is_valid_testcase': validation['classification']['is_valid_testcase'],
            'is_genuine_bug': validation['classification']['is_genuine_bug'],
            'recommendation': validation['classification']['recommendation'],
            'confidence': validation['classification']['confidence'],
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))
    
    classification = validation['classification']
    print(f"\n📊 Validation Summary:")
    print(f"   Valid Test Case: {'✅' if classification['is_valid_testcase'] else '❌'}")
    print(f"   Genuine Bug: {'✅' if classification['is_genuine_bug'] else '❌'}")
    print(f"   Recommendation: {classification['recommendation']}")
    print(f"   Confidence: {classification['confidence']}")
    
    if classification['reasoning']:
        print(f"\n   Reasoning:")
        for reason in classification['reasoning'][:3]:
            print(f"   - {reason}")
    
    print(f"\n📁 Output files:")
    print(f"   {validation_json}")
    print(f"   {validation_md}")
    print(f"\n   Next step: python3 check_duplicates.py {workdir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
