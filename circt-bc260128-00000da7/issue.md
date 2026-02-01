# [arcilator] Timeout with dynamic-indexed packed struct array in always_comb

## Description
`arcilator` hangs indefinitely (times out) when processing a SystemVerilog `always_comb` block that contains a write followed by a read to the same dynamic index of an unpacked array of packed structs. 

Industry-standard tools like Verilator and Slang compile this code successfully, confirming it is valid SystemVerilog. The issue appears to be a flaw in `arcilator`'s dependency analysis or combinational loop detection, specifically when dealing with dynamic array indexing and packed struct fields.

## Minimized Reproduction Code
```systemverilog
// bug.sv
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
# This command hangs indefinitely
circt-verilog --ir-hw bug.sv | arcilator
```

With a timeout:
```bash
timeout 60s bash -c 'circt-verilog --ir-hw bug.sv | arcilator'
```

## Actual Behavior
`arcilator` enters an infinite loop or fails to resolve a perceived combinational cycle, leading to a compilation timeout.

## Expected Behavior
`arcilator` should either:
1. Successfully compile the code, recognizing that writing to `arr[idx].data` and reading from `arr[idx].valid` are independent field accesses even if the index is dynamic.
2. If it cannot prove independence, it should handle the dependency without entering an infinite loop.

## Root Cause Analysis Summary
The bug is suspected to be in `ConvertToArcs` or `LowerState` passes where combinational loops are detected and handled. 
1. **Dynamic Indexing**: The use of a runtime input `idx` prevents the compiler from statically determining which array element is accessed, forcing it to treat all elements as potentially conflicting.
2. **Coarse-Grained Dependency Tracking**: `arcilator` likely treats a write to `arr[idx].data` as a write to the entire `arr[idx]` element. When it subsequently sees a read from `arr[idx].valid`, it perceives a dependency cycle.
3. **Loop Detection Failure**: The dependency graph for the `always_comb` block seems to form a cycle that the tool's loop detection/resolution logic cannot handle gracefully, resulting in an infinite loop during compilation.

Suspected components:
- `lib/Conversion/ConvertToArcs/ConvertToArcs.cpp` (Loop detection logic around line 180)
- `lib/Dialect/Arc/Transforms/LowerState.cpp` (Combinational loop handling around line 286)

## Environment
- **CIRCT Version**: 1.139.0
- **Toolchain**: LLVM 22
- **OS**: Linux

## Additional Context
Similar patterns with static indices do not seem to trigger the timeout, suggesting the dynamic index is a key factor in the analysis failure. 

No closely matching existing issues were found. Related but distinct issues include #9469 (focuses on `always_ff` sensitivity lists).
