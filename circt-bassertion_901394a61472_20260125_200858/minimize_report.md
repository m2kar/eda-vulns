# Minimized test case for CIRCT arcilator crash

## Minimal Reproduction Command
```bash
circt-verilog --ir-hw bug.sv | arcilator
```

## Size Reduction
- Original: 24 lines (source.sv)
- Minimized: 6 lines (bug.sv)
- Reduction: 75%

## Key Insight
The crash is triggered by a single `inout` (bidirectional) port.
All other constructs (struct, arrays, always_ff, etc.) are not required.

## IR Generated
```mlir
module {
  hw.module @MinimalInout(in %c : !llhd.ref<i1>) {
    hw.output
  }
}
```

The `inout logic c` becomes `in %c : !llhd.ref<i1>`, and arcilator's
LowerState pass cannot create a StateType for the `!llhd.ref<i1>` type.
