# [circt-verilog][arcilator] sim.fmt.literal legalization failure with immediate assertion in always_comb

## Description

`arcilator` fails to lower `sim.fmt.literal` operations generated from immediate assertions (`assert ... else $error`) in `always_comb` blocks. The operation is marked as legal in `LowerArcToLLVM` pass but lacks proper conversion patterns, causing legalization failures.

## Reproduction

### Minimal Testcase

```systemverilog
module m(input x);
  always_comb assert (x) else $error("f");
endmodule
```

### Command

```bash
circt-verilog --ir-hw bug.sv | arcilator
```

### Error Output

```
<stdin>:3:10: error: failed to legalize operation 'sim.fmt.literal'
    %0 = sim.fmt.literal "Error: f"
         ^
<stdin>:3:10: note: see current operation: %7 = "sim.fmt.literal"() <{literal = "Error: f"}> : () -> !sim.fstring
```

### Generated IR (before arcilator)

```mlir
module {
  hw.module @m(in %x : i1) {
    %0 = sim.fmt.literal "Error: f"
    llhd.combinational {
      cf.cond_br %x, ^bb2, ^bb1
    ^bb1:
      sim.proc.print %0
      cf.br ^bb2
    ^bb2:
      llhd.yield
    }
    hw.output
  }
}
```

## Root Cause Analysis

### Issue

The `sim::FormatLiteralOp` created from the `$error` message becomes an orphaned operation in the `LowerArcToLLVM` pass:

1. `circt-verilog` converts the SystemVerilog immediate assertion to Moore IR
2. `MooreToCore` pass transforms `moore::FormatLiteralOp` → `sim::FormatLiteralOp`
3. In `arcilator`'s `LowerArcToLLVM.cpp`:
   - Line 1087: `sim::FormatLiteralOp` is marked as **legal**
   - Lines 809-822: Format operations should be consumed by `SimPrintFormattedProcOpLowering` pattern
   - However, in combinational contexts (`always_comb`), the `sim.fmt.literal` operation may not be properly connected to a `PrintFormattedProcOp`
   - Result: Orphaned operation triggers legalization failure

### Suspected Locations

| File | Lines | Description |
|------|-------|-------------|
| `lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp` | L1087 | Marks sim.fmt.* as legal |
| `lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp` | L809-822 | FormatLiteralOp handling |
| `lib/Dialect/Sim/Transforms/ProceduralizeSim.cpp` | L112 | Checks non-FormatLiteralOp |
| `lib/Conversion/MooreToCore/MooreToCore.cpp` | L1998-2005 | FormatLiteralOp conversion |

### Code Path

```
circt-verilog (ImportVerilog)
    ↓ $error("...")
moore::FormatLiteralOp
    ↓ MooreToCore pass
sim::FormatLiteralOp
    ↓ arcilator pipeline
[MISSING] Assert message not properly bound to PrintFormattedOp
    ↓ LowerArcToLLVM
❌ Orphaned sim.fmt.literal causes legalization failure
```

## Validation

| Tool | Valid | Notes |
|------|-------|-------|
| circt-verilog | ✅ Yes | Successfully parsed and generated MLIR IR |
| Verilator | ✅ Yes | Warning with `-Wall`, passes without |
| Slang | ✅ Yes | Parse successful |

## Classification

- **Type**: Legalization failure (incomplete support)
- **Dialect**: SystemVerilog / Arc / Sim
- **Impact**: `always_comb` blocks with immediate assertions containing formatted error messages

## Related Issues

- #9395 - [circt-verilog][arcilator] Arcilator assertion failure (CLOSED, more general)
- #8286 - [circt-verilog][llhd][arcilator] Verilog-to-LLVM lowering issues (OPEN, LLVM lowering)
- #6810 - [Arc] Add basic assertion support (OPEN, feature request)

## Possible Fixes

1. **Properly connect format ops**: Ensure `sim.fmt.literal` in combinational contexts is connected to appropriate consumer ops
2. **Cleanup unused ops**: Remove orphaned `sim.fmt.literal` operations before legalization
3. **Add explicit support**: Implement proper handling for immediate assertions in `always_comb` blocks
4. **Early rejection**: Add check to reject or warn about unsupported constructs earlier in the pipeline

## Environment

- **CIRCT Version**: 1.139.0
- **LLVM Version**: 22.0.0git
- **Tools**: circt-verilog, arcilator, opt, llc

## Original Testcase

```systemverilog
module test_module(
  input logic clk,
  output logic [7:0] arr
);
  logic [2:0] idx;

  always_ff @(posedge clk) begin
    idx <= idx + 1;
  end

  always_comb begin
    arr = 8'b0;
    arr[idx] = 1'b1;
    assert (arr[idx] == 1'b1) else $error("Assertion failed: arr[%0d] != 1", idx);
  end
endmodule
```

**Reduction**: 22 lines → 3 lines (84.7% compression)
