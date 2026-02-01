# Root Cause Analysis Report

## Crash Summary

| Property | Value |
|----------|-------|
| **Crash Type** | Timeout (60s) |
| **Crash Hash** | 27a7406d01d3 |
| **Dialect** | SystemVerilog |
| **Tool Chain** | circt-verilog → arcilator → opt → llc |
| **CIRCT Version** | 1.139.0 |

## Test Case Analysis

### Source Code (`source.sv`)

```systemverilog
module M(
  input  logic A,
  output logic O
);

  // Packed struct declaration with multiple fields
  typedef struct packed {
    logic field1;
    logic field2;
  } my_struct_t;

  // Packed struct variable declaration
  my_struct_t my_struct;

  // Wire for conditional
  logic C;

  // Assignment to packed struct field
  always_comb begin
    my_struct.field2 = 1'b0;
  end

  // Member access to packed struct field used in conditional
  assign C = my_struct.field1;

  // Conditional statement (if-else) inside always block
  always_comb begin
    if (C)
      O = A;
    else
      O = 1'b1;
  end

endmodule
```

### Key Constructs Identified

1. **Packed struct type definition** (`my_struct_t`) with two 1-bit fields
2. **Partial struct field assignment** in `always_comb` - only `field2` is assigned
3. **Struct field read** - `field1` is read but never explicitly assigned
4. **Conditional logic** depending on the unassigned struct field

## Root Cause Hypothesis

### Primary Cause: Combinational Loop in Struct Field Access

The timeout is most likely caused by **false combinational loop detection** in the arcilator lowering pipeline. The issue occurs due to:

1. **Partial Struct Assignment Pattern**: The test case only assigns `my_struct.field2 = 1'b0` while reading `my_struct.field1`. This creates a pattern where:
   - `hw.struct_inject` operations are generated for partial field updates
   - The inject operation reads the current struct value to update a single field
   - This creates a cyclic dependency in the dataflow graph

2. **Missing Canonicalization for Partial Struct Updates**: Based on CIRCT issue #8860, the `hw.struct_inject` operation has canonicalizers for fully-defined struct assignments but may not handle partial assignments optimally. When not all fields are assigned, the inject chain creates a circular reference.

3. **LowerState Pass Infinite Loop**: The `LowerState.cpp` pass uses a worklist-based depth-first traversal to lower operations. When it encounters an operation on a combinational loop, it may enter an infinite loop during the `opsWorklist` processing, particularly in the `lowerOp` method.

### Supporting Evidence

1. **Issue #6373**: Arcilator explicitly does not support `hw.struct` types in `arc.tap` operations, indicating incomplete struct support in the Arc dialect.

2. **Issue #8286**: Documents known issues with Moore to LLVM lowering, including problems with combinational logic containing control flow operators.

3. **Issue #8860**: Documents the exact issue with array/struct element assignments creating false combinational loops. The key insight is:
   > "The Mem2Reg pass converts array element assignments into `hw.array_inject`s. But there is no canonicalizer that detects if a chain of injects fully specify the array..."
   
   For structs, while there is a canonicalizer for fully-defined structs, partial assignments remain problematic.

### Execution Path Analysis

```
1. circt-verilog --ir-hw source.sv
   └── Converts SV to HW dialect
   └── Packed struct becomes hw.struct type
   └── Partial field assignment becomes hw.struct_inject

2. arcilator (HANGS HERE)
   └── LowerState pass attempts to lower hw.module to arc.model
   └── Detects cyclic dependency in struct inject operations
   └── Worklist processing enters infinite loop
```

## Technical Details

### The Problematic Pattern

When lowering the partial struct assignment:
```systemverilog
always_comb begin
  my_struct.field2 = 1'b0;
end
```

The MLIR IR likely generates:
```mlir
// Read current struct value
%current = ... // current value of my_struct
// Inject field2 with constant 0
%updated = hw.struct_inject %current["field2"], %c0 : !hw.struct<field1: i1, field2: i1>
// The updated value becomes the new my_struct
```

The issue is that `%current` needs `%updated` (as the struct is self-referential in comb logic), creating a cycle.

### Why Timeout Instead of Error

The `LowerState` pass in `LowerState.cpp` has cycle detection:
```cpp
if (!opsSeen.insert({defOp, phase}).second) {
  defOp->emitOpError("is on a combinational loop");
  dumpWorklist();
  return failure();
}
```

However, this detection only works if the same `{op, phase}` pair is encountered again. In cases where the cycle involves different phases or complex interdependencies, the worklist may grow indefinitely without triggering this check, leading to a timeout rather than an explicit error.

## Severity Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Reproducibility** | High | Deterministic timeout with specific pattern |
| **Impact** | Medium | Blocks compilation of valid SystemVerilog code |
| **Workaround** | Possible | Fully initialize struct fields |
| **Fix Complexity** | Medium | Requires canonicalizer enhancement or cycle detection improvement |

## Recommendations

1. **Immediate Workaround**: Initialize all struct fields explicitly:
   ```systemverilog
   always_comb begin
     my_struct.field1 = 1'b0; // Add explicit initialization
     my_struct.field2 = 1'b0;
   end
   ```

2. **Potential Fix Areas**:
   - Enhance `hw.struct_inject` canonicalizer to handle partial assignments
   - Improve cycle detection in `LowerState` pass to catch this pattern earlier
   - Add timeout/iteration limit in the worklist processing

## Related Issues

- **CIRCT #6373**: [Arc] Support hw.wires of aggregate types
- **CIRCT #8286**: [circt-verilog][llhd][arcilator] Verilog-to-LLVM lowering issues  
- **CIRCT #8860**: [LLHD] Assigning array elements individually creates a combinational loop

## Files Analyzed

- `error.txt`: Timeout error message
- `source.sv`: Test case source code
- `LowerState.cpp`: CIRCT source for Arc LowerState pass (via GitHub)
