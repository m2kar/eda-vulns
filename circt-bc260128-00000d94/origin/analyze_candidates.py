#!/usr/bin/env python3
import json
import re

# Load scored issues
with open('scored_issues.json', 'r') as f:
    scored_issues = json.load(f)

# Load error.txt to get exact assertion
with open('error.txt', 'r') as f:
    error_content = f.read()

# Extract key signatures
current_assertion = "dyn_cast on a non-existent value"
current_crash_location = "sanitizeInOut"
current_function = "getModulePortInfo"
current_file = "MooreToCore.cpp"
current_port_type = "string output port"  # From test case

print("=" * 70)
print("DETAILED DUPLICATE ANALYSIS")
print("=" * 70)
print()
print(f"Current Crash Signature:")
print(f"  - Assertion: {current_assertion}")
print(f"  - Crash Location: {current_crash_location}")
print(f"  - Function: {current_function}")
print(f"  - Root Issue: String type conversion in module port")
print()

# Analyze top 3
top_issues = scored_issues[:3]

for issue in top_issues:
    number = issue['number']
    title = issue['title']
    score = issue['similarity_score']
    body = (issue.get('body') or '').lower()
    url = issue['url']
    
    print(f"Issue #{number}: {title}")
    print(f"  Score: {score:.1f}/20")
    print(f"  URL: {url}")
    print()
    
    # Check for crash signature match
    has_dyn_cast = 'dyn_cast' in body and 'non-existent' in body
    has_mooretocore = 'mooretocore' in body
    has_assertion = 'assertion' in body
    has_string = 'string' in body
    has_conversion = 'conversion' in body
    has_port = 'port' in body
    
    print(f"  Crash Signature Match:")
    print(f"    - dyn_cast assertion: {has_dyn_cast}")
    print(f"    - MooreToCore pass: {has_mooretocore}")
    print(f"    - String type: {has_string}")
    print(f"    - Type conversion: {has_conversion}")
    print(f"    - Port-related: {has_port}")
    print()
    
    # Analysis
    if number == 8930:
        print(f"  Analysis: Different root cause")
        print(f"    - Same dyn_cast assertion failure pattern")
        print(f"    - BUT: Crash in sqrt/floor conversion (real type)")
        print(f"    - NOT in port handling (sanitizeInOut)")
        print(f"    - Different symptom: sqrt/floor vs string port")
        print(f"    - Related but DISTINCT issues")
    elif number == 8332:
        print(f"  Analysis: Highly relevant")
        print(f"    - Directly discusses StringType support")
        print(f"    - Focus: sim to llvm lowering strategy")
        print(f"    - Not a crash report, but feature request")
        print(f"    - Related: StringType support missing")
    elif number == 8283:
        print(f"  Analysis: Very relevant")
        print(f"    - String variable compilation failure")
        print(f"    - moore.variable with string type cannot be legalized")
        print(f"    - Focus: MooreToCore string type conversion")
        print(f"    - Related: Missing string type support in conversion")
    
    print()
    print("-" * 70)
    print()

print()
print("=" * 70)
print("FINAL RECOMMENDATION")
print("=" * 70)
print()
print("Recommendation: review_existing")
print("Confidence: HIGH")
print()
print("Reasoning:")
print("  1. #8930 shows same dyn_cast assertion but different location")
print("  2. #8332 & #8283 discuss missing StringType support")
print("  3. Current crash: String port in module (getModulePortInfo)")
print("  4. All issues relate to MooreToCore string/type conversion")
print("  5. This appears to be a known limitation: StringType handling")
print()
print("Action Items:")
print("  - Check if #8332 or #8283 already tracks string port issue")
print("  - If not, create new issue (separate from sqrt/floor crash)")
print("  - Cross-reference related issues in bug report")

