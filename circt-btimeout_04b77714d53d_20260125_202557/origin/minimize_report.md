# Minimization Report

## Summary

| Metric | Value |
|--------|-------|
| **Original file** | source.sv (35 lines) |
| **Minimized file** | bug.sv (5 lines) |
| **Reduction** | 85.7% |
| **Timeout preserved** | Yes |

## Original Test Case

```systemverilog
module M #(parameter DEPTH=16) (
  input  logic [1:0] a,
  output logic [3:0] z
);
  typedef struct packed {
    logic [DEPTH-1:0] data;
    logic valid;
  } mystruct_t;

  mystruct_t struct_in;

  always_comb begin
    struct_in.data = {a, {DEPTH-2{1'b0}}};
  end

  my_module #(.DEPTH(DEPTH)) inst (
    .D_flopped(struct_in)
  );

  assign z = struct_in.valid ? struct_in.data[3:0] : 4'b0;
endmodule

module my_module #(parameter DEPTH=16) (
  input logic [DEPTH:0] D_flopped
);
endmodule
```

## Minimized Test Case

```systemverilog
module M (output logic z);
  struct packed { logic a; logic b; } s;
  always_comb s.a = 1;
  assign z = s.b;
endmodule
```

## Minimization Process

### Phase 1: Initial Analysis
- Identified key components from analysis.json
- Original hypothesis: implicit struct-to-logic type coercion at module port

### Phase 2: Module Reduction
- Removed `my_module` and its instantiation
- **Result**: Timeout still occurs → Module instantiation NOT required

### Phase 3: Parameter Removal
- Removed `#(parameter DEPTH=16)`
- Changed struct field width from parameterized to fixed
- **Result**: Timeout still occurs → Parameters NOT required

### Phase 4: Port Simplification
- Reduced input from `logic [1:0] a` to `logic a`, then removed entirely
- Reduced output from `logic [3:0] z` to `logic z`
- **Result**: Timeout still occurs

### Phase 5: Struct Simplification
- Reduced struct from `{logic [DEPTH-1:0] data; logic valid}` to `{logic a; logic b}`
- Removed `typedef`
- **Result**: Timeout still occurs → Struct size irrelevant

### Phase 6: Assignment Simplification
- Simplified always_comb from multi-expression to single constant assignment
- **Result**: Timeout still occurs

### Phase 7: Verification of Essential Components

| Component | Required | Notes |
|-----------|----------|-------|
| `always_comb` assigning struct field | **YES** | Using `assign` instead does NOT cause timeout |
| Reading different struct field | **YES** | Reading same field or no read does NOT cause timeout |
| Output port using read value | **YES** | Without output, dead code elimination removes problematic pattern |

## Root Cause Refinement

The original hypothesis was **incorrect**. The actual bug pattern is:

**always_comb block assigning to one struct field + reading a different struct field in continuous assignment**

This causes the MooreToHW conversion to enter an infinite loop or non-terminating state during:
1. Partial struct field assignment handling in `always_comb`
2. Data flow analysis when different fields are read vs written

The module instantiation and type coercion identified in the original analysis were NOT contributing factors.

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv
```

## Files Generated

- `bug.sv` - Minimized test case (5 lines)
- `error.log` - Timeout log
- `command.txt` - Reproduction command
