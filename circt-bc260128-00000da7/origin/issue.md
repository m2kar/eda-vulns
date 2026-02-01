# [arcilator] Timeout with dynamic-indexed packed struct array in always_comb

## Description

`arcilator` enters an infinite loop (timeout) when processing `always_comb` blocks that contain write-then-read patterns on the same dynamic index of a packed struct array.

The issue occurs when:
- An unpacked array of packed structs is defined
- In an `always_comb` block, one field of a dynamically-indexed array element is written
- A different field of the same dynamically-indexed array element is read

The dependency analysis in `arcilator` appears to incorrectly treat writes to different fields of the same struct element as creating a combinational loop, causing the tool to hang indefinitely.

## Reproduction Code

```systemverilog
// Minimized testcase: arcilator timeout on dynamic-indexed packed struct array
// Bug: always_comb with write-then-read on same dynamic index causes infinite loop

module bug(
  input logic [2:0] idx,
  output logic result
);

  typedef struct packed {
    logic valid;
    logic [7:0] data;
  } elem_t;

  elem_t arr [0:7];

  always_comb begin
    arr[idx].data = 8'hFF;       // Write to .data field
    result = arr[idx].valid;     // Read from .valid field (same index)
  end

endmodule
```

## Reproduction Command

```bash
# Reproduce the timeout (hangs indefinitely)
/opt/firtool/bin/circt-verilog --ir-hw bug.sv | /opt/firtool/bin/arcilator

# With timeout (completes with exit code 124)
timeout 60s bash -c '/opt/firtool/bin/circt-verilog --ir-hw bug.sv | /opt/firtool/bin/arcilator'
echo "Exit code: $?"  # Expected: 124
```

## Actual Behavior

`arcilator` hangs indefinitely (timeout after 300s in the original report, or 60s with the test command).

**Observed:**
- `circt-verilog --ir-hw` successfully generates HW dialect IR
- `arcilator` enters an infinite loop during dependency analysis
- No error message is emitted; the process just hangs

## Expected Behavior

The code should compile successfully without timeout.

**Expected:**
- `arr[idx].data` and `arr[idx].valid` should be recognized as independent fields of the same packed struct element
- Writing to `.data` should not block reading from `.valid` in the same combinational block
- The dependency analysis should not detect a false combinational loop

## Root Cause Analysis

The issue is likely in the dependency analysis passes of `arcilator`, specifically:

1. **ConvertToArcs** (`lib/Conversion/ConvertToArcs/ConvertToArcs.cpp:180-183`):
   - The combinational loop detection may treat accesses to different fields of the same dynamically-indexed struct element as the same operation
   - This creates a false dependency cycle: `write(arr[idx]) -> read(arr[idx])`

2. **LowerState** (`lib/Dialect/Arc/Transforms/LowerState.cpp:286-290`):
   - The worklist-based dependency resolution may add the same operation infinitely
   - The `opsSeen` check may fail to detect the loop when dealing with field-level dependencies

**Hypothesis:**
`arcilator` lacks field-level dependency tracking for packed structs. It treats `arr[idx].data` (write) and `arr[idx].valid` (read) as accesses to the entire `arr[idx]` element, creating a false combinational loop that cannot be resolved.

## Environment

- **CIRCT Version**: 1.139.0
- **LLVM Version**: 22
- **Platform**: Linux
- **Reproducible**: Yes

## Validation

The test case has been validated as valid SystemVerilog:

- **Verilator**: ✅ Compiles without errors or warnings
- **Slang**: ✅ Compiles without errors or warnings
- **CIRCT**: ❌ Times out during `arcilator` processing

This confirms the issue is specific to CIRCT's dependency analysis, not a problem with the SystemVerilog syntax.

## Related Issues

- #9469: `[circt-verilog][arcilator] Inconsistent compilation behavior: direct array indexing in always_ff sensitivity list vs. intermediate wire`
  - Different issue (always_ff vs always_comb, sensitivity list vs combinational loop)
  - Low similarity score: 4.0/15

No identical issues found in the CIRCT issue tracker.

## Additional Notes

**Workarounds:**
- Using separate always_comb blocks for writes and reads may avoid the false loop detection
- Using static indices instead of dynamic indices avoids the issue
- Breaking the packed struct into separate variables works

**Potential Fixes:**
1. Implement field-level dependency tracking for packed structs
2. Add iteration limits to dependency analysis passes to prevent infinite loops
3. Improve dynamic indexing handling in dependency resolution
4. Add timeout protection with meaningful error messages when a loop is detected
