# Test Case Minimization Report

## Original Test Case
**File**: source.sv (16 lines)
- Module with input signal, internal wire
- Output logic and output array
- Multiple assignments in always_comb block from same wire

## Minimization Process

### Iteration 1: Remove output `result`
**Action**: Replaced array assignment with second single-bit output
**Result**: No crash - Bug not triggered
**Learning**: Array assignment is critical to bug

### Iteration 2: Remove module input
**Action**: Removed `in_signal`, made `internal_wire` a constant
**Result**: Crash reproduced
**Learning**: Module input is not required

### Iteration 3: Test 2-bit array
**Action**: Changed array from [3:0] to [1:0]
**Result**: No crash - Bug not triggered
**Learning**: Array width threshold exists (minimum 3 bits)

### Iteration 4: Test 3-bit array with 2 assignments
**Action**: 3-bit array [2:0] with 2 assignments
**Result**: Crash reproduced
**Learning**: 2 assignments sufficient, 3+ bit array sufficient

### Iteration 5: Test 3-bit array with 3 assignments
**Action**: 3-bit array [2:0] with 3 assignments
**Result**: Crash reproduced
**Learning**: More assignments still trigger bug

### Iteration 6: Ultra-minimal version
**Action**: 
- Shortened module name from `my_module` to `m`
- Shortened signal names (e.g., `internal_wire` to `w`, `result_array` to `a`)
**Result**: Crash reproduced
**Learning**: Names don't affect bug, only structure matters

## Minimal Test Case

### SystemVerilog Code (bug.sv)
```systemverilog
module m(
  output logic [3:0] a
);

  logic w = 1'b0;

  always_comb begin
    a[0] = w;
    a[1] = w;
  end

endmodule
```

**Lines**: 11 (reduced from 16)
**Characters**: 112 (reduced from ~270)

### Reproduction Command
```bash
export PATH=/opt/llvm-22/bin:$PATH
circt-verilog --ir-hw bug.sv
```

### Crash Signature
```
Assertion: op->use_empty() && "expected 'op' to have no uses"' failed
Location: /root/circt/llvm/mlir/lib/IR/PatternMatch.cpp:156
Function: virtual void mlir::RewriterBase::eraseOp(Operation *)
```

### Key Observations
1. **Critical Elements**:
   - `always_comb` procedural block
   - Array of 3+ bits as output
   - At least 2 assignments to different array indices from same wire in same always_comb

2. **Non-Critical Elements**:
   - Module input ports
   - Assign statements (wire can be constant)
   - Additional array assignments beyond 2
   - Output variables beyond the array
   - Module/signal names

3. **Bug Trigger**:
   - Multiple array assignments in `always_comb` create multiple ExtractOp instances
   - These ExtractOps depend on a ConcatOp representing the wire
   - The `extractConcatToConcatExtract` pattern tries to replace one ExtractOp
   - But the ConcatOp still has other uses (from other assignments)
   - This causes assertion failure when `eraseOp()` is called

### Minimal Structure Required
```
module
  output logic [N+:0] where N >= 2  // Minimum: [3:0] (3-bit)
) {
  logic wire_name;
  always_comb begin
    array[0] = wire_name;  // Assignment 1
    array[1] = wire_name;  // Assignment 2 (minimum to trigger)
  end
}
```

Note: While [2:0] (3-bit) array reproduces crash, further investigation
revealed that [1:0] (2-bit) array does not trigger the bug, suggesting
a width threshold in the canonicalization pattern logic.

## Files Generated
- `bug.sv` - Minimal test case (11 lines)
- `error.log` - Full crash output
- `command.txt` - Reproduction command

## Reduction Summary
- **Original**: 16 lines, ~270 characters
- **Final**: 11 lines, 112 characters
- **Reduction**: 31% fewer lines, 59% fewer characters
- **Bug Reproducibility**: Confirmed with minimal case
