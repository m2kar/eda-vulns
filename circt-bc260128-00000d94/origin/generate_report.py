#!/usr/bin/env python3
import json
import datetime

# Load data
with open('analysis.json', 'r') as f:
    analysis = json.load(f)

with open('scored_issues.json', 'r') as f:
    scored_issues = json.load(f)

# Extract top issues
top_10 = scored_issues[:10]
top_score = top_10[0]['similarity_score'] if top_10 else 0

# Determine recommendation
if top_score >= 10.0:
    recommendation = "review_existing"
    confidence = "high"
    reason = "High similarity score (>10) indicates this may be related to existing issues"
elif top_score >= 5.0:
    recommendation = "likely_new"
    confidence = "medium"
    reason = "Related issues exist but this appears to be a variant or new manifestation"
else:
    recommendation = "new_issue"
    confidence = "high"
    reason = "No similar issues found in the repository"

# Generate duplicates.json
report_json = {
    "version": "1.0",
    "timestamp": datetime.datetime.now().isoformat(),
    "case_id": "260128-00000d94",
    "crash_signature": {
        "assertion": analysis.get('assertion_message', ''),
        "crash_location": analysis.get('crash_location', {}),
        "failing_pass": analysis.get('failing_pass', ''),
        "dialect": analysis.get('dialect', '')
    },
    "search_strategy": {
        "keywords": analysis.get('keywords', []),
        "search_terms": [
            "StringType",
            "DynamicStringType", 
            "getModulePortInfo",
            "sanitizeInOut",
            "MooreToCore",
            "type conversion"
        ]
    },
    "results": {
        "total_issues_found": len(scored_issues),
        "top_10_similar": [
            {
                "number": issue['number'],
                "title": issue['title'],
                "url": issue['url'],
                "state": issue['state'],
                "similarity_score": issue['similarity_score'],
                "search_label": issue.get('search_label', '')
            }
            for issue in top_10
        ]
    },
    "recommendation": {
        "action": recommendation,
        "confidence": confidence,
        "top_score": top_score,
        "top_issue": {
            "number": top_10[0]['number'] if top_10 else None,
            "title": top_10[0]['title'] if top_10 else None,
            "url": top_10[0]['url'] if top_10 else None,
            "reason": "Most similar: Same MooreToCore pass, type conversion issues, related to StringType handling"
        },
        "reason": reason
    },
    "analysis": {
        "issue_8930": {
            "relationship": "RELATED but DISTINCT",
            "description": "Same dyn_cast assertion, but different crash location (sqrt/floor vs string port)",
            "severity": "Related pattern but separate bug"
        },
        "issue_8332": {
            "relationship": "RELATED FEATURE REQUEST",
            "description": "Feature request for StringType support in Moore to LLVM lowering",
            "severity": "Missing feature that affects string type handling"
        },
        "issue_8283": {
            "relationship": "HIGHLY RELEVANT",
            "description": "String type variable compilation failure in MooreToCore",
            "severity": "Known limitation with string type conversion"
        }
    }
}

# Save JSON report
with open('duplicates.json', 'w') as f:
    json.dump(report_json, f, indent=2)

print("✓ duplicates.json generated")

# Generate duplicates.md
markdown_content = f"""# Duplicate Check Report

**Case ID**: {report_json['case_id']}  
**Timestamp**: {report_json['timestamp']}

## Crash Signature

| Property | Value |
|----------|-------|
| Assertion | {analysis.get('assertion_message', 'N/A')} |
| Pass | {analysis.get('failing_pass', 'N/A')} |
| Dialect | {analysis.get('dialect', 'N/A')} |
| File | {analysis.get('crash_location', {}).get('file', 'N/A')} |
| Function | {analysis.get('crash_location', {}).get('function', 'N/A')} |

## Search Results

**Total Issues Found**: {len(scored_issues)}  
**Top Similarity Score**: {top_score:.1f}/20

## Top 10 Similar Issues

"""

for i, issue in enumerate(top_10, 1):
    markdown_content += f"""### {i}. #{issue['number']} (Score: {issue['similarity_score']:.1f}) [{issue['state']}]

**Title**: {issue['title']}

**URL**: {issue['url']}

---

"""

markdown_content += f"""## Recommendation

**Action**: `{recommendation}`

**Confidence**: {confidence}

**Reason**: {reason}

"""

if recommendation == "review_existing":
    markdown_content += """
### ⚠️ Review Required

A highly similar issue (#8332 or #8283) was found. This appears to be related to **missing StringType support in MooreToCore conversion**.

**Key Findings**:
- Issue #8930: Same assertion pattern but different location (sqrt/floor vs string port)
- Issue #8332: Feature request for StringType Moore->LLVM lowering  
- Issue #8283: String variable compilation failure in MooreToCore

**Recommended Next Steps**:
1. Review issue #8332 and #8283 to check if they already cover string port conversion
2. If this is a distinct manifestation (string OUTPUT port vs string variable), proceed with new report
3. If already covered, consider marking as duplicate with cross-reference
4. Otherwise, create new issue referencing related issues

"""

elif recommendation == "likely_new":
    markdown_content += """
### 📋 Proceed with Caution

Related issues exist in the repository but this appears to be a different or related variant.

**Recommended Next Steps**:
1. Proceed to generate the full bug report
2. Reference related issues in the bug description
3. Highlight specific differences from existing issues

"""

else:
    markdown_content += """
### ✅ Clear to Proceed

No identical or highly similar issues found in the repository.

**Recommended Next Steps**:
1. Proceed to generate and submit the complete bug report
2. This appears to be a new issue

"""

markdown_content += f"""## Detailed Issue Analysis

### Issue #8930: [MooreToCore] Crash with sqrt/floor

**Relationship**: RELATED but DISTINCT

**Similarity**: Same dyn_cast assertion failure  
**Difference**: Crash in sqrt/floor conversion (real type) vs string port handling  
**Verdict**: Different root cause - keep separate

---

### Issue #8332: [MooreToCore] Support for StringType from moore to llvm dialect

**Relationship**: HIGHLY RELEVANT - Feature Request

**Description**: Discussion about adding StringType support from Moore to LLVM dialect  
**Status**: Open feature request, not a crash report  
**Connection**: Addresses the broader issue of StringType handling

---

### Issue #8283: [ImportVerilog] Cannot compile forward decleared string type

**Relationship**: HIGHLY RELEVANT - Same Root Issue

**Description**: String variable (moore.variable with string type) cannot be legalized in MooreToCore  
**Status**: Open bug  
**Connection**: Directly related to StringType conversion failure in MooreToCore

---

## Scoring Weights Used

| Factor | Weight | Match |
|--------|--------|-------|
| Title keyword match | 2.0 | Per keyword found |
| Body keyword match | 1.0 | Per keyword found |
| Assertion message match | 3.0 | If present |
| Dialect label match | 1.5 | If matches |
| Failing pass match | 2.0 | If mentions pass |
| MooreToCore specific | 2.0 | If mentions module |
| Type conversion | 1.5 | If mentions conversion |
| Port/module related | 1.0 | If port-related |
| String type | 1.5 | If string mentioned |
| Dyn_cast/assertion | 1.0 | If crash-related |

## Summary

This crash is part of a **broader StringType handling limitation in MooreToCore conversion**. Multiple related issues exist but they appear to address different manifestations:

- String variables in procedures (Issue #8283)
- String type to LLVM lowering strategy (Issue #8332)
- Real type conversion assertion (Issue #8930)
- **This case**: String output port in module signature

Consider this a **new but related issue** that should be reported with cross-references to existing issues.
"""

with open('duplicates.md', 'w') as f:
    f.write(markdown_content)

print("✓ duplicates.md generated")
print()
print("=" * 70)
print("DUPLICATE CHECK COMPLETE")
print("=" * 70)
print(f"Recommendation: {recommendation}")
print(f"Top Score: {top_score:.1f}/20")
print(f"Top Issue: #{top_10[0]['number']} - {top_10[0]['title']}")
print("=" * 70)

