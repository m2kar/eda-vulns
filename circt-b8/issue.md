## Title

[circt-verilog][hw] `--ir-hw` timeout in canonicalize after `llhd-sig2reg` on cyclic `hw.struct_inject`/`hw.bitcast`

## Summary

`circt-verilog --ir-hw` hangs (timeout) on a minimal valid SystemVerilog testcase.

The timeout is inside `circt-verilog` LLHD->core lowering (not in `arcilator`).

## Reproducer

```systemverilog
module M(output logic O);
  typedef struct packed { logic a; logic b; } S;
  S s;
  always_comb s.b = 0;
  assign O = s.a;
endmodule
```

## Commands

```bash
# Hangs
timeout 20s circt-verilog --ir-hw bug.sv

# Control: succeeds
timeout 20s circt-verilog --ir-llhd bug.sv
```

## Actual vs Expected

- Actual: `--ir-hw` times out (exit code `124`, no diagnostics).
- Expected: `--ir-hw` should terminate and emit HW/core IR (or emit a proper diagnostic), but not hang.

## Where It Hangs

Pass bisection of the LLHD->core sequence shows first timeout at:

- `canonicalize{...}` right after `llhd-sig2reg, cse`

## Evidence (IR before failing canonicalize)

```mlir
%1 = hw.bitcast %3 : (!hw.struct<a: i1, b: i1>) -> i2
%2 = hw.bitcast %1 : (i2) -> !hw.struct<a: i1, b: i1>
%3 = hw.struct_inject %2["b"], %false : !hw.struct<a: i1, b: i1>
%a = hw.struct_extract %2["a"] : !hw.struct<a: i1, b: i1>
```

This creates a cycle: `%3 -> %1 -> %2 -> %3`.

## Suspected Root Cause

Non-converging canonicalization on cyclic aggregate update pattern involving `hw.struct_inject` + `hw.bitcast` produced after `llhd-sig2reg`.

Likely relevant code:

- `tools/circt-verilog/circt-verilog.cpp`
- `lib/Dialect/HW/HWOps.cpp` (`StructInjectOp::canonicalize`, `BitcastOp::canonicalize`, `StructExtractOp::canonicalize`)

## Environment

- CIRCT: `firtool-1.139.0`
- LLVM: `22.0.0git`
- slang: `9.1.0+0`
- OS: Linux x86_64
