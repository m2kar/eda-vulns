# CIRCT Bug Report

## Summary
`circt-verilog` → `arcilator` → `opt` → `llc` fails during legalization with
`failed to legalize operation 'sim.fmt.literal'` when an `$error` assertion is
present. The legalization failure occurs in the Arc-to-LLVM conversion path.

## Steps to Reproduce

### Minimal Testcase (`bug.sv`)
```systemverilog
module test_module;
  logic q;

  always_comb begin
    q = 1'b1;
  end

  always @(*) begin
    assert (q == 1'b0) else $error("Assertion failed: q != 0");
  end
endmodule
```

### Command
```bash
circt-verilog --ir-hw bug.sv | arcilator | opt -O0 | llc -O0 --filetype=obj -o /tmp/circt_260129_0000129c_min.o
```

## Actual Result
```
<stdin>:3:10: error: failed to legalize operation 'sim.fmt.literal'
    %0 = sim.fmt.literal "Error: Assertion failed: q != 0"
         ^
<stdin>:3:10: note: see current operation: %0 = "sim.fmt.literal"() <{literal = "Error: Assertion failed: q != 0"}> : () -> !sim.fstring
```

## Expected Result
The pipeline should either lower `sim.fmt.literal` or mark it legal for the
Arc-to-LLVM conversion, so the compilation proceeds without legalization
failure.

## Root Cause Analysis (Summary)
`sim.fmt.literal` is a Sim dialect op (see `include/circt/Dialect/Sim/SimOps.td`).
The Arc-to-LLVM conversion (`lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp`) does
not legalize or lower `sim::Format*` ops. When `$error` introduces
`sim.fmt.literal`, it survives into legalization and fails.

## Environment
* CIRCT: firtool-1.139.0 (from `/opt/firtool/bin`)
* LLVM: 22.0.0git (from `/opt/llvm-22/bin`)
* OS: Linux (container)

## Duplicate Check
No exact match found. Related issues with different illegal ops:
* #9467 (arcilator legalization failure on `llhd.constant_time`)
* #8286 (general arcilator lowering issues)
