# Root Cause Analysis: sim.fmt.literal Legalization Failure

## Summary

**Crash Type**: Legalization Failure  
**Dialect**: sim (Simulation Dialect)  
**Failed Operation**: `sim.fmt.literal`  
**Tool**: arcilator  
**Severity**: High - blocks simulation of assertions with format strings

## Error Context

```
<stdin>:3:10: error: failed to legalize operation 'sim.fmt.literal'
    %0 = sim.fmt.literal "Error: Assertion failed: q != 0"
         ^
<stdin>:3:10: note: see current operation: 
  %3 = "sim.fmt.literal"() <{literal = "Error: Assertion failed: q != 0"}> : () -> !sim.fstring
```

## Triggering Code Pattern

The crash is triggered by a SystemVerilog immediate assertion with `$error` system function:

```systemverilog
always @(*) begin
  assert (q == 1'b0) else $error("Assertion failed: q != 0");
end
```

### Key Elements:
1. **Immediate assertion** (`assert ... else ...`)
2. **`$error` system function** with format string argument
3. **Format string literal**: `"Assertion failed: q != 0"`

## Tool Pipeline Analysis

```
circt-verilog --ir-hw → arcilator → opt → llc
                         ↑
                     CRASH HERE
```

1. `circt-verilog --ir-hw`: Parses SystemVerilog, generates HW IR
   - `$error("...")` is lowered to `sim.fmt.literal` operation
2. `arcilator`: Attempts to lower HW IR for simulation
   - **FAILS**: No legalization pattern for `sim.fmt.literal`

## Root Cause

**arcilator's legalization pass lacks a conversion pattern for `sim.fmt.literal` operation.**

When the sim dialect's format string literal operation (`sim.fmt.literal`) is generated from SystemVerilog's `$error` system function, arcilator cannot lower it to its target representation (LLVM IR via arc dialect).

### Technical Details:

- The `sim.fmt.literal` operation produces a `!sim.fstring` type
- This operation represents a static format string used in simulation messages
- arcilator needs to either:
  1. Implement a lowering pattern to handle `sim.fmt.literal` 
  2. Or strip/ignore these operations if simulation messages are not supported

## Affected SystemVerilog Constructs

Any assertion or simulation construct using format strings:
- `$error("message")`
- `$fatal("message")`
- `$warning("message")`
- `$info("message")`
- `$display("message")` (potentially)

## Minimal Reproducer

```systemverilog
module test;
  logic q = 1;
  always @(*) begin
    assert (q == 0) else $error("assertion failed");
  end
endmodule
```

## Suggested Fix Areas

1. **lib/Dialect/Arc/Transforms/Legalization** - Add pattern for `sim.fmt.literal`
2. **lib/Dialect/Sim/SimOps.td** - Check if lowering interface is defined
3. **include/circt/Conversion/SimToArc.h** - May need new conversion pass

## References

- CIRCT Sim Dialect: https://circt.llvm.org/docs/Dialects/Sim/
- Arcilator documentation: https://circt.llvm.org/docs/Tools/arcilator/
