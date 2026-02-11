# Root Cause Analysis Report

## Executive Summary

The previous conclusion was incorrect: this timeout is **not** in `arcilator` LowerState.

Fresh reproduction shows the hang already happens in:

- `circt-verilog --ir-hw bug.sv`

So the failure is in `circt-verilog` LLHD->core lowering, specifically at a later canonicalization stage.

## Reproduction Evidence

### 1) `--ir-hw` hangs

Command:

```bash
/usr/bin/timeout 20s /opt/firtool/bin/circt-verilog --ir-hw bug.sv
```

Observed:

- Exit code: `124`
- No diagnostics printed

### 2) `--ir-llhd` succeeds

Command:

```bash
/usr/bin/timeout 20s /opt/firtool/bin/circt-verilog --ir-llhd bug.sv
```

Observed:

- Exit code: `0`
- LLHD IR emitted successfully

This localizes the issue to LLHD->core work performed for `--ir-hw`.

## Pass-Level Localization

Using the pass sequence reported by:

```bash
circt-verilog --ir-hw --verbose-pass-executions bug.sv
```

and replaying the LLHD->core sequence incrementally with `circt-opt`, the first timeout appears at:

- `canonicalize{...}` after `llhd-sig2reg, cse`

All earlier steps complete.

## IR Shape Right Before Timeout

IR snapshot immediately before the failing canonicalize:

```mlir
%1 = hw.bitcast %3 : (!hw.struct<a: i1, b: i1>) -> i2
%2 = hw.bitcast %1 : (i2) -> !hw.struct<a: i1, b: i1>
%3 = hw.struct_inject %2["b"], %false : !hw.struct<a: i1, b: i1>
%a = hw.struct_extract %2["a"] : !hw.struct<a: i1, b: i1>
```

This forms a self-cycle in SSA use/def graph:

- `%3 -> %1 -> %2 -> %3`

## Root Cause Hypothesis (Updated)

High confidence: non-converging canonicalization over a cyclic aggregate update form.

Pipeline effect:

1. `always_comb` partial field assignment lowers to `hw.struct_inject`.
2. `llhd-sig2reg` materializes a cyclic pattern involving `hw.struct_inject` and `hw.bitcast`.
3. Subsequent HW canonicalization fails to converge in practical time on that pattern and times out.

## Relevant Code Areas

- `tools/circt-verilog/circt-verilog.cpp` (pipeline construction/order)
- `lib/Dialect/HW/HWOps.cpp`
  - `StructInjectOp::canonicalize`
  - `BitcastOp::canonicalize`
  - `StructExtractOp::canonicalize`

## Why the Previous Report Was Wrong

The old report attributed the timeout to `arcilator` LowerState.

But since `circt-verilog --ir-hw` alone already times out, the failure occurs **before** any `arcilator | opt | llc` stage can run.

## Impact and Workaround

### Impact

- Valid SystemVerilog can cause frontend compilation hang.
- Deterministic reproducer; high reproducibility.

### Workaround

Avoid partial packed-struct updates in this pattern; initialize all fields in the same `always_comb` block.

## Suggested Fix Directions

1. Add cycle-aware safeguards for `struct_inject`/`bitcast` SCC patterns in HW canonicalization.
2. Add canonicalization/folding to break or normalize self-referential `struct_inject` update forms emitted after `llhd-sig2reg`.
3. Add regression test ensuring `circt-verilog --ir-hw` terminates on this reproducer.

## Environment

- CIRCT: `firtool-1.139.0`
- LLVM: `22.0.0git`
- slang: `9.1.0+0`
- OS: Linux x86_64
