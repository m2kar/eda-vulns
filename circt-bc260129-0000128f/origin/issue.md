# arcilator assertion in InferStateProperties with array state

## Summary
`arcilator` asserts in `InferStateProperties` when processing a design that
writes to an array in a variable-bound loop and reads from that array. The pass
attempts to create a `hw.constant` with a non-integer type, triggering a
`cast<mlir::IntegerType>` assertion. On the current toolchain, the same input
fails earlier with a verifier error on `arc.state` operand types.

## Steps to Reproduce
Original command (from crash log):
```
circt-verilog --ir-hw source.sv | arcilator | opt -O0 | llc -O0 --filetype=obj -o /tmp/featurefuzz_sv_xfhtv50x/test_8c1820bbc9c1.o
```

Minimized command (current toolchain):
```
circt-verilog --ir-hw bug.sv | arcilator | opt -O0 | llc -O0 --filetype=obj -o /tmp/arcilator_repro.o
```

## Actual Behavior
- **Original toolchain**: assertion failure
  - `cast<Ty>() argument of incompatible type!`
  - Location: `lib/Dialect/Arc/Transforms/InferStateProperties.cpp:211`
- **Current toolchain**: verifier error
  - `'arc.state' op operand type mismatch: operand #2`

## Expected Behavior
`arcilator` should handle or reject the array-typed state cleanly without
asserting. If unsupported, it should emit a user-facing diagnostic instead of
crashing.

## Crash Signature (original)
```
arcilator: /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/Casting.h:566: decltype(auto) llvm::cast(From &) [To = mlir::IntegerType, From = mlir::Type]: Assertion `isa<To>(Val) && "cast<Ty>() argument of incompatible type!"' failed.
```

## Stack Trace (excerpt)
```
applyEnableTransformation(circt::arc::DefineOp, circt::arc::StateOp, ...)
  /lib/Dialect/Arc/Transforms/InferStateProperties.cpp:211:55
InferStatePropertiesPass::runOnStateOp
  /lib/Dialect/Arc/Transforms/InferStateProperties.cpp:454:17
```

## Minimized Testcase (bug.sv)
```systemverilog
module example_module(
  input logic clk,
  input logic [3:0] length,
  input logic [31:0] data [0:15],
  output logic q
);

  logic [31:0] data_reg [0:15];

  always_ff @(posedge clk) begin
    for (int i = 0; i < length; i++) begin
      data_reg[i] <= data[i];
    end
  end

  always_ff @(posedge clk) begin
    q <= data_reg[0];
  end

endmodule
```

## Environment
- circt-verilog: LLVM 22.0.0git; CIRCT firtool-1.139.0; slang 9.1.0+0
- arcilator/opt/llc from `/opt/firtool/bin`

## Notes
The Arc `InferStateProperties` pass appears to assume integer-typed values when
constructing constants. With array-typed state (`!hw.array<16xi32>`), this leads
to either a verifier error or an assertion depending on toolchain version.
