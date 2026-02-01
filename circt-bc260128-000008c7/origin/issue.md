# [CIRCT][Arcilator] InferStateProperties assertion on arc.state with array-of-struct type

## Summary
`arcilator` hits an assertion in `llvm::cast<mlir::IntegerType>` during
`InferStateProperties` when processing an `arc.state` derived from a packed
array-of-struct SystemVerilog type. The original crash is an assertion in
`HW::ConstantOp::create`. On the current toolchain the pipeline fails earlier
with an `arc.state` operand type mismatch verifier error.

## Reproducer

### Testcase (bug.sv)
```systemverilog
module test_module (
  input logic clk,
  input logic rst,
  output logic valid
);

  typedef struct packed {
    logic [7:0] data;
    logic [3:0] id;
  } pkt_t;

  pkt_t [3:0] packet_array;

  always_ff @(posedge clk) begin
    if (rst) begin
      valid <= 1'b0;
      for (int i = 0; i < 4; i++) begin
        packet_array[i].id = i;
        packet_array[i].data = 8'b0;
      end
    end else begin
      valid <= |packet_array[0].data;
    end
  end

endmodule
```

### Reproduction Command
```bash
circt-verilog --ir-hw source.sv | arcilator | opt -O0 | llc -O0 --filetype=obj -o /tmp/repro_260128-000008c7.o
```

## Observed Behavior

### Original Crash (from error.txt)
```
arcilator: /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/Casting.h:566: decltype(auto) llvm::cast(From &) [To = mlir::IntegerType, From = mlir::Type]: Assertion `isa<To>(Val) && "cast<Ty>() argument of incompatible type!"' failed.
```

### Current Toolchain Output
```
<stdin>:17:11: error: 'arc.state' op operand type mismatch: operand #1
    %10 = comb.mux bin %rst, %6, %packet_array : !hw.array<4xstruct<data: i8, id: i4>>
          ^
<stdin>:17:11: note: see current operation: %2 = "arc.state"(%4, %arg1, %0, %1) <{arc = @test_module_arc, latency = 1 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 2, 0>}> : (!seq.clock, i1, i1, i382917464) -> !hw.array<4xstruct<data: i8, id: i4>>
<stdin>:17:11: note: expected type: '!hw.array<4xstruct<data: i8, id: i4>>'
<stdin>:17:11: note:   actual type: 'i382917464'
```

## Expected Behavior
The pipeline should either accept the aggregate `arc.state` operand type or
gracefully report an error without triggering assertions in `InferStateProperties`.

## Root Cause Analysis (summary)
`InferStateProperties` appears to assume `IntegerType` when creating constants.
The testcase drives an `arc.state` with an aggregate type (`hw.array<struct<...>>`),
leading to an invalid cast and assertion in the original crash.

## Environment
* Toolchain: circt-verilog/arcilator from `/opt/firtool/bin`
* LLVM tools: `/opt/llvm-22/bin`
* Reproducer and logs are in `origin/`:
  * `bug.sv`, `command.txt`, `error.log`, `reproduce.log`, `root_cause.md`, `analysis.json`

## Additional Notes
The current toolchain fails earlier during verification with an operand type mismatch,
which may be related to the same underlying assumption about integer-only state types.
