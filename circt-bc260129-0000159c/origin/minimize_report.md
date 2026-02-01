# Minimize Report

## Summary
- **Original file**: source.sv (313 bytes, 20 lines)
- **Minimized file**: bug.sv (32 bytes, 2 lines)
- **Reduction**: 89.8%

## Minimization Process

### Iteration 1: Remove for loop and temp_reg
- **Action**: Simplified sequential logic, removed for loop and multi-bit register
- **Result**: ✅ Crash still reproduced

```sv
// test_v1.sv
module MixedPorts(
  input  logic clk,
  input  logic a,
  output logic b,
  inout  logic c
);
  logic reg1;
  always_ff @(posedge clk) begin
    reg1 <= a;
  end
  assign b = reg1;
  assign c = (a) ? reg1 : 1'bz;
endmodule
```

### Iteration 2: Remove tri-state assignment and output port
- **Action**: Removed output port `b` and tri-state assignment to `c`
- **Result**: ✅ Crash still reproduced

```sv
// test_v2.sv
module Test(
  input  logic clk,
  input  logic a,
  inout  logic c
);
  logic reg1;
  always_ff @(posedge clk) begin
    reg1 <= a;
  end
endmodule
```

### Iteration 3: Minimal always_ff
- **Action**: Simplified sequential logic to minimal form
- **Result**: ✅ Crash still reproduced

```sv
// test_v3.sv
module Test(
  input logic clk,
  inout logic c
);
  logic r;
  always_ff @(posedge clk) r <= 0;
endmodule
```

### Iteration 4: Remove sequential logic entirely
- **Action**: Removed all internal logic, keeping only inout port
- **Result**: ✅ Crash still reproduced

```sv
// test_v4.sv
module Test(inout logic c);
endmodule
```

### Iteration 5: Use implicit type
- **Action**: Removed explicit `logic` type, use minimal port declaration
- **Result**: ✅ Crash still reproduced

```sv
// test_v6.sv (final)
module Test(inout c);
endmodule
```

## Final Minimized Test Case

```sv
module Test(inout c);
endmodule
```

## Key Findings

1. **Essential Trigger**: Only `inout` port declaration is required to trigger the crash
2. **Sequential Logic NOT Required**: The `always_ff` block mentioned in analysis.json as "secondary" is actually not necessary - the bug triggers on any module with inout port
3. **Root Cause Confirmed**: The arcilator's LowerState pass attempts to create `arc.state` storage for inout ports, which are represented as `llhd.ref<T>` types. The `StateType::verify()` fails because `computeLLVMBitWidth()` doesn't handle `RefType`.

## Crash Signature Match
- **Error Message**: `state type must have a known bit width; got '!llhd.ref<i1>'`
- **Crash Location**: `LowerState.cpp:219` → `StateType::get()` → assertion failure
- **Same as Original**: ✅ Yes

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv | arcilator
```

With full path:
```bash
/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw bug.sv | /edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/arcilator
```
