# [circt-verilog] Infinite loop when processing always_comb with self-referential packed struct output

## Description

`circt-verilog` hangs indefinitely when compiling SystemVerilog code that contains:
1. A packed struct as an output port
2. An `always_comb` block that reads one field and writes another field of the same output struct

The tool enters an infinite loop without producing any output or error message.

## Minimal Reproducer

```systemverilog
typedef struct packed { logic a; logic b; } s_t;

module top(output s_t out);
  always_comb begin
    out.b = out.a;  // Read from out.a, write to out.b
  end
endmodule
```

## Steps to Reproduce

```bash
circt-verilog --ir-hw bug.sv
# Command never terminates
```

## Expected Behavior

The compiler should either:
1. Successfully compile the code (the fields `out.a` and `out.b` are distinct), or
2. Report a diagnostic about the combinational dependency/loop

## Actual Behavior

`circt-verilog` hangs indefinitely. The process must be killed manually (e.g., via timeout or Ctrl+C).

## Cross-Tool Validation

| Tool | Result | Notes |
|------|--------|-------|
| **Slang** | ✅ Pass | Valid SystemVerilog syntax |
| **Verilator** | ⚠️ Warning | Detects combinational loop: `UNOPTFLAT: Circular combinational logic: 'out'` |
| **CIRCT** | ❌ Hang | Infinite loop |

Verilator correctly identifies that there's a combinational loop issue, but CIRCT should at minimum report this rather than hanging.

## Environment

- **CIRCT Version**: 1.139.0
- **LLVM Version**: 22.0.0git
- **OS**: Linux

## Analysis

The bug appears to be triggered by the combination of:
1. **Packed struct output port** - Using an array (`logic [1:0] out`) instead of a struct does NOT trigger the hang
2. **always_comb block** - Using `assign` (continuous assignment) instead does NOT trigger the hang
3. **Self-referential field access** - Reading `out.a` while writing `out.b` within the same always_comb block

### Key Finding
The packed struct type seems essential to trigger the bug. The same pattern with a vector/array type compiles successfully:

```systemverilog
// This works fine:
module top(output logic [1:0] out);
  always_comb begin
    out[1] = out[0];  // No hang
  end
endmodule
```

## Related Issues

- #3403 - Circular logic not detected (related to MLIR-level circular logic)
- #8022 - Infinite loop in OrOp folder (different trigger, different stage)
- #9057 - Topological cycle after importing verilog (related but different symptom)

## Labels
`bug`, `ImportVerilog`, `Verilog/SystemVerilog`
