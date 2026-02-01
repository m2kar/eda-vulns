# Minimization Report

## Test Case ID
260128-00000883

## Original Test Case (source.sv)
- **Lines**: 24
- **Characters**: 505

```systemverilog
typedef struct packed {
  logic [7:0] value;
} data_struct_t;

typedef union packed {
  data_struct_t s;
  logic [7:0] raw;
} data_union_t;

module data_processor(
  input logic clk,
  input data_union_t in_data,
  output logic [7:0] result
);

  function logic [7:0] process_data(data_union_t du);
    return du.s.value ^ du.raw;
  endfunction

  always_ff @(posedge clk) begin
    result <= process_data(in_data);
  end

endmodule
```

## Minimized Test Case (bug.sv)
- **Lines**: 6
- **Characters**: 64

```systemverilog
typedef union packed {
  logic [7:0] a;
} u_t;

module m(input u_t i);
endmodule
```

## Reduction Statistics
- **Line reduction**: 24 → 6 (75% reduction)
- **Character reduction**: 505 → 64 (87% reduction)

## What Was Removed and Why

### 1. Nested Struct Definition (data_struct_t)
- **Removed**: The `typedef struct packed { logic [7:0] value; } data_struct_t;`
- **Reason**: The crash is triggered by the packed union type conversion, not the nested struct. A simple `logic [7:0]` field in the union is sufficient.

### 2. Unnecessary Ports
- **Removed**: `input logic clk`, `output logic [7:0] result`
- **Reason**: Only the packed union input port triggers the bug. Clock and output are unrelated to the type conversion failure.

### 3. Function Definition (process_data)
- **Removed**: The entire `function logic [7:0] process_data(data_union_t du); ... endfunction`
- **Reason**: The crash occurs during module port conversion, before any function processing. Functions are not involved in triggering the bug.

### 4. Always Block Logic
- **Removed**: The entire `always_ff @(posedge clk) begin ... end`
- **Reason**: Sequential logic is not involved in the type conversion crash which happens during MooreToCore pass at module port processing.

### 5. Multiple Union Members
- **Removed**: Second union member (`logic [7:0] raw`)
- **Reason**: A single-member packed union is sufficient to trigger the null type conversion.

### 6. Simplified Names
- **Changed**: `data_union_t` → `u_t`, `in_data` → `i`, `data_processor` → `m`
- **Reason**: Shorter names for readability; names don't affect the bug trigger.

## Essential Elements Preserved
1. **typedef union packed**: The packed union type definition
2. **Module with packed union port**: `input u_t i` - the exact construct that causes the type converter to return null

## Crash Signature Verification
Both original and minimized test cases produce the same crash:
- **Assertion**: `dyn_cast on a non-existent value`
- **Location**: `PortImplementation.h:177` in `sanitizeInOut()`
- **Root cause function**: `getModulePortInfo()` at `MooreToCore.cpp:259`
