# Root Cause Analysis Report

## Executive Summary

The CIRCT compilation pipeline times out after 60 seconds when processing a SystemVerilog module that passes a packed struct type to a module port expecting plain packed logic. The type coercion between `mystruct_t` (packed struct) and `logic [DEPTH:0]` during module instantiation likely triggers inefficient handling in the arcilator pass, causing non-terminating or exponentially slow compilation.

## Crash Context

- **Tool/Command**: `circt-verilog --ir-hw | arcilator | opt -O0 | llc -O0`
- **Dialect**: Moore (SystemVerilog frontend) → HW/Arc
- **Failing Stage**: Likely `arcilator` pass or HW dialect lowering
- **Crash Type**: Timeout (60s limit exceeded)
- **CIRCT Version**: 1.139.0

## Error Analysis

### Error Message
```
Compilation timed out after 60s
```

### Pipeline
```
circt-verilog --ir-hw → arcilator → opt -O0 → llc -O0
```

The timeout occurs during the CIRCT pipeline, most likely in:
1. `circt-verilog --ir-hw` conversion from SystemVerilog to HW IR, or
2. `arcilator` simulation model generation

## Test Case Analysis

### Code Summary
A parameterized module `M` defines a packed struct `mystruct_t` containing a data field and valid bit. This struct is instantiated and passed to a submodule `my_module` whose port type is plain packed logic rather than the struct type.

### Key Constructs

| Construct | Description | Relevance to Timeout |
|-----------|-------------|---------------------|
| `typedef struct packed` | Defines `mystruct_t` with `logic [DEPTH-1:0] data` + `logic valid` | Packed struct to logic conversion |
| Parameterized module | `DEPTH=16` propagates to struct width | Width-dependent operations |
| Struct-to-logic port | `struct_in` passed to `logic [DEPTH:0]` port | **Type coercion trigger** |
| Array slicing | `struct_in.data[3:0]` | Additional type complexity |
| Replication operator | `{a, {DEPTH-2{1'b0}}}` | Compile-time expansion |

### Source Code

```systemverilog
module M #(parameter DEPTH=16) (
  input  logic [1:0] a,
  output logic [3:0] z
);

  // Struct type definition
  typedef struct packed {
    logic [DEPTH-1:0] data;
    logic valid;
  } mystruct_t;

  // Struct signal
  mystruct_t struct_in;

  // Assignment from multi-bit input to struct field
  always_comb begin
    struct_in.data = {a, {DEPTH-2{1'b0}}};
  end

  // Instantiate parameterized module with struct input
  my_module #(.DEPTH(DEPTH)) inst (
    .D_flopped(struct_in)  // <-- PROBLEMATIC: struct passed to logic port
  );

  // Assignment from struct to multi-bit output
  assign z = struct_in.valid ? struct_in.data[3:0] : 4'b0;

endmodule

// Parameterized module definition
module my_module #(parameter DEPTH=16) (
  input logic [DEPTH:0] D_flopped  // DEPTH bits data + 1 bit valid = 17 bits
);
  // Module implementation (placeholder)
endmodule
```

### Potentially Problematic Patterns

1. **Struct-to-plain-logic type mismatch at port binding**:
   - `struct_in` is type `mystruct_t` (packed struct, 17 bits total)
   - `D_flopped` is type `logic [16:0]` (plain packed, 17 bits)
   - While bit-compatible, this requires implicit type conversion

2. **Packed struct field access combined with conditional**:
   - `struct_in.valid ? struct_in.data[3:0] : 4'b0`
   - Requires both field extraction and array slicing

3. **Parameterized width in struct definition**:
   - The struct field width depends on module parameter

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence): Arcilator Struct-to-Logic Lowering Loop

**Cause**: The arcilator pass enters an inefficient or non-terminating state when attempting to lower struct-typed signals that are connected to plain logic ports.

**Evidence**:
- The module instantiation `my_module #(.DEPTH(DEPTH)) inst (.D_flopped(struct_in))` connects a struct to a plain logic port
- Arcilator needs to generate simulation code that handles this type coercion
- The conversion between aggregate types (struct) and flat bit-vectors may trigger repeated lowering attempts or exponential IR expansion

**Mechanism**: 
1. Moore dialect converts SystemVerilog to HW IR, preserving struct types
2. Arcilator attempts to lower the struct-to-logic connection
3. Without proper bitcast handling, it may repeatedly try to decompose/recompose the struct
4. Each iteration adds more intermediate operations, leading to timeout

### Hypothesis 2 (Medium Confidence): HW IR Type Coercion Explosion

**Cause**: The `circt-verilog --ir-hw` conversion generates inefficient IR when handling struct-to-packed-logic implicit conversions, creating excessive intermediate operations.

**Evidence**:
- The conversion happens at the Moore-to-HW boundary
- SystemVerilog allows implicit struct-to-packed conversion at module ports
- The HW dialect may need explicit operations to represent this

**Mechanism**:
1. During Moore-to-HW conversion, the struct assignment to logic port requires bitcast
2. The IR generation may create excessive operations for struct decomposition
3. This causes downstream passes (arcilator) to slow down exponentially

### Hypothesis 3 (Lower Confidence): Parameter Elaboration with Struct Types

**Cause**: The combination of parameterized module instantiation with struct types causes elaboration complexity.

**Evidence**:
- Both modules are parameterized with DEPTH
- The struct type depends on the parameter
- Double elaboration (parent and child module parameters) with struct types

**Mechanism**:
The parameter-dependent struct type combined with cross-module port binding creates complex type checking or elaboration that scales poorly.

## Suggested Fix Directions

1. **For Users (Workaround)**:
   - Explicitly cast struct to packed bits before port connection:
     ```systemverilog
     my_module #(.DEPTH(DEPTH)) inst (
       .D_flopped({struct_in.data, struct_in.valid})  // Explicit decomposition
     );
     ```

2. **For CIRCT Developers**:
   - Add efficient bitcast operation for struct-to-packed-logic conversions in arcilator
   - Implement short-circuit handling when aggregate types are bit-compatible with flat types
   - Add detection for non-terminating lowering patterns with timeout/error

3. **Investigation Points**:
   - Profile arcilator on this testcase to identify hot loops
   - Check if hw.bitcast is properly generated and lowered
   - Verify struct lowering passes have termination guarantees

## Keywords for Issue Search

`arcilator` `struct` `timeout` `packed` `type conversion` `bitcast` `moore` `hw` `non-terminating` `port` `module instantiation`

## Related Files to Investigate

- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Struct lowering to HW types
- `lib/Dialect/Arc/Transforms/` - Arcilator transformation passes
- `lib/Dialect/HW/Transforms/` - HW dialect type handling
- `include/circt/Dialect/Moore/MooreTypes.td` - Moore struct type definitions

## Severity Assessment

| Aspect | Rating |
|--------|--------|
| Impact | **Medium-High** - Blocks compilation of valid SystemVerilog |
| Reproducibility | **High** - Deterministic timeout with specific pattern |
| Workaround | **Available** - Explicit type casting |
| Priority | **Medium** - Struct-to-logic port binding is common pattern |

## Conclusion

The timeout is most likely caused by inefficient handling of packed struct types being passed to module ports expecting plain packed logic. The arcilator pass (or preceding HW lowering) likely enters a slow or non-terminating state when attempting to generate simulation code for this type conversion. The root issue is insufficient optimization or early termination handling for aggregate-to-flat type coercions in the CIRCT compilation pipeline.
