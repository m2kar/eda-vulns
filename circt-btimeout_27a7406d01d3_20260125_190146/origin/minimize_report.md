# Minimization Report

## Summary
Successfully minimized the test case from 34 lines to 6 lines while preserving the bug-triggering behavior.

## Original Test Case (source.sv)
- **Lines of code**: 34
- **File size**: 622 bytes
- **Key constructs**: 
  - Packed struct with 2 fields (field1, field2)
  - Partial field assignment in always_comb (only field2)
  - Read of unassigned field (field1)
  - Conditional logic dependent on unassigned field
  - Separate intermediate wire for conditional

## Minimized Test Case (bug.sv)
- **Lines of code**: 6
- **File size**: 123 bytes
- **Key constructs preserved**:
  - Packed struct typedef with 2 fields (a, b)
  - Partial field assignment in always_comb (s.b = 0)
  - Read of unassigned field (s.a)
  - Direct output assignment

## Reduction Statistics
- **Line reduction**: 34 → 6 lines (82.4% reduction)
- **Size reduction**: 622 → 123 bytes (80.2% reduction)

## Minimization Process

### Step 1: Initial Analysis
Read analysis.json to identify critical constructs:
1. Packed struct typedef with multiple fields
2. Partial struct field assignment (only one field assigned)
3. Read of unassigned struct field
4. always_comb block (essential - continuous assign does NOT trigger bug)

### Step 2: First Minimization Attempt
Reduced to 6 lines preserving all critical constructs:
```systemverilog
module M(input logic A, output logic O);
  typedef struct packed { logic f1; logic f2; } S;
  S s;
  always_comb s.f2 = 1'b0;
  assign O = s.f1 ? A : 1'b1;
endmodule
```
**Result**: Timeout at 30s ✓

### Step 3: Further Reduction
Removed input port and simplified output logic:
```systemverilog
module M(output logic O);
  typedef struct packed { logic a; logic b; } S;
  S s;
  always_comb s.b = 0;
  assign O = s.a;
endmodule
```
**Result**: Timeout at 30s ✓

### Step 4: Verification
Tested replacing `always_comb` with continuous `assign`:
```systemverilog
module M(output logic O);
  typedef struct packed { logic a, b; } S;
  S s;
  assign s.b = 0;  // Using assign instead of always_comb
  assign O = s.a;
endmodule
```
**Result**: Compiles successfully (NO timeout)

This confirms `always_comb` is essential for triggering the bug.

### Step 5: Final Verification
Confirmed bug.sv times out at 60s threshold as expected.

## Key Findings
1. The bug is triggered by partial packed struct field assignment **inside always_comb**
2. Continuous assignment (`assign`) does NOT trigger the bug
3. Reading any unassigned field of the struct triggers the timeout
4. The bug is in the arcilator pipeline, specifically the LowerState pass

## Conclusion
The minimal test case successfully preserves the essential bug-triggering pattern while removing all extraneous code. The 82% line reduction makes the bug easier to analyze and report.
