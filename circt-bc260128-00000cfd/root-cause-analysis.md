# Root Cause Analysis: CIRCT Bug 260128-00000cfd

## Summary
**Test Case**: SystemVerilog assertion with array indexing causes `sim.fmt.literal` legalization failure in arcilator
**Error Message**: `failed to legalize operation 'sim.fmt.literal'` during ArcToLLVM conversion
**Severity**: Compilation failure - arcilator cannot complete conversion to LLVM

## Error Details

### Original Command
```bash
circt-verilog --ir-hw test.sv | arcilator | opt -O0
```

### Error Output
```
<stdin>:6:10: error: failed to legalize operation 'sim.fmt.literal'
    %0 = sim.fmt.literal "Error: Assertion failed: arr["
         ^
<stdin>:6:10: note: see current operation: %56 = "sim.fmt.literal"() <{literal = "Error: Assertion failed: arr["}> : () -> !sim.fstring
```

## IR Before ArcToLLVM Conversion

```mlir
module {
  hw.module @test_module(in %clk : i1, out result : i8) {
    %c0_i8 = hw.constant 0 : i8
    %c6_i8 = hw.constant 6 : i8
    %c1_i3 = hw.constant 1 : i3
    %0 = sim.fmt.literal "Error: Assertion failed: arr["
    %1 = sim.fmt.literal "] != 1"
    %c1_i8 = hw.constant 1 : i8
    %c-1_i3 = hw.constant -1 : i3
    llhd.combinational {
      %6 = comb.sub %c-1_i3, %idx : i3
      %7 = hw.array_get %arr[%6] : !hw.array<8xi8>, i3
      %8 = comb.icmp eq %7, %c1_i8 : i8
      cf.cond_br %8, ^bb2, ^bb1
    ^bb1:  // pred: ^bb0
      %9 = sim.fmt.dec %idx specifierWidth 0 : i3
      %10 = sim.fmt.concat (%0, %9, %1)
      sim.proc.print %10
      cf.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llhd.yield
    }
    %2 = comb.sub %c-1_i3, %idx : i3
    %3 = hw.array_inject %arr[%2], %c0_i8 : !hw.array<8xi8>
    %4 = comb.add %idx, %c1_i3 : i3
    %5 = seq.to_clock %clk
    %idx = seq.compreg %4, %5 : i3
    %arr = seq.compreg %3, %5 : !hw.array<8xi8>
    hw.output %c6_i8 : i8
  }
}
```

**Key observation**: `sim.fmt.literal`, `sim.fmt.dec`, `sim.fmt.concat`, and `sim.proc.print` operations are created by `circt-verilog` from SystemVerilog `$error()` system task with format string.

## Root Cause

### Primary Issue: Format Operations Not Properly Lowered in ArcToLLVM

**File**: `/home/zhiqing/edazz/circt/lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp`

The ArcToLLVM conversion framework:
1. **Marks format operations as LEGAL** (lines 1241-1243):
   ```cpp
   target.addLegalOp<sim::FormatLiteralOp, sim::FormatDecOp, sim::FormatHexOp,
                     sim::FormatBinOp, sim::FormatOctOp, sim::FormatCharOp,
                     sim::FormatStringConcatOp>();
   ```
   This means they don't need direct conversion to LLVM ops.

2. **Has `SimPrintFormattedProcOpLowering` pattern** (lines 874-902):
   - Calls `foldFormatString(rewriter, op.getInput(), stringCache)` to walk format tree
   - Replaces `sim.proc.print` with runtime call
   - Format operations are NOT directly removed by this pattern

3. **Problem**: Format operations survive through the conversion despite being marked legal, because:
   - They are expected to become dead code after print op is replaced
   - But CSE pass runs AFTER ArcToLLVM, too late to clean them up
   - Format operations inside `arc.execute` regions may have scoping issues

### Why Conversion Fails

The `arc.execute` lowering pattern (lines 906-951) **inlines** the body region:
```cpp
rewriter.inlineRegionBefore(op.getBody(), blockAfter);
```

This causes operations inside `arc.execute` (including `sim.fmt.*` and `sim.proc.print`) to be moved into the surrounding block context. However:

1. **Type Conversion Issue**: `FormatStringType` is converted to `LLVM::LLVMPointerType` (line 1263):
   ```cpp
   converter.addConversion([&](sim::FormatStringType type) {
     return LLVM::LLVMPointerType::get(type.getContext());
   });
   ```
   This creates pointer without specifying pointee type, which may cause legalization issues.

2. **Region Inlining Context**: When `sim.proc.print` is inlined from `arc.execute`, the conversion framework may not properly handle the scoping of format operations that reference module-level constants.

3. **Missing Dead Code Elimination**: The conversion assumes format operations will be eliminated as dead code, but the CSE pass runs after conversion, not during it.

### Supporting Evidence

**Git History**: Commit f40496973 (Jan 19, 2026) added nascent support for sim.proc.print and sim.fmt.* operations. This suggests support was recently added and may be incomplete or buggy.

**Code Location**: The `SimPrintFormattedProcOpLowering` pattern exists in the conversion and is registered (line 1318), but it appears to not successfully handle all cases where format operations are inside `arc.execute` regions.

## Technical Details

### Operation Definitions

**`sim.fmt.literal`** (SimOps.td:155-171):
- Creates constant string fragments for formatted printing
- Arguments: `StrAttr:$literal`
- Results: `FormatStringType:$result`
- Traits: `[Pure, ConstantLike]`
- Has folder: true

**`sim.proc.print`** (SimOps.td:501-512):
- Prints formatted string within procedural region
- Arguments: `FormatStringType:$input`
- No direct lowering to LLVM

### Conversion Pipeline Order

1. Preprocessing: `populateArcPreprocessingPipeline`
   - Strip OM, Emit, SV dialects
   - Lower FIR memories
   - Lower Verif simulations
   - Add taps
   - Strip SV
   - Infer memories
   - Lower DPI functions
   - CSE + canonicalize

2. Arc Conversion: `populateArcConversionPipeline`
   - Convert to arcs
   - Dedup
   - Flatten modules
   - CSE + canonicalize

3. Arc Optimization: `populateArcOptimizationPipeline`
   - Split loops
   - Dedup
   - Infer state properties
   - CSE
   - Merge taps
   - Make LUTs
   - CSE + canonicalize

4. Arc State Lowering: `populateArcStateLoweringPipeline`
   - Lower state
   - Inline arcs
   - Merge ifs
   - CSE + canonicalize

5. Arc State Allocation: `populateArcStateAllocationPipeline`
   - Lower arcs to funcs
   - Allocate state
   - Lower clocks to funcs
   - Split funcs
   - CSE + canonicalize

6. **ArcToLLVM**: `populateArcToLLVMPipeline` - **FAILURE POINT**
   - Convert bitcasts
   - Insert runtime
   - **Lower Arc to LLVM** - format ops should be processed here
   - CSE
   - Arc canonicalize

## Conclusion

The root cause is that **format operations inside `arc.execute` regions fail to be properly handled during ArcToLLVM conversion**. The conversion framework expects these operations to be eliminated as dead code after the print operation is replaced, but they remain in the IR and cause legalization failures when the LLVM dialect conversion framework tries to verify all operations are legal.

This is likely a **missing feature or bug** in how the ArcToLLVM pass handles format operations that appear in the context of arc.execute regions, particularly those created from SystemVerilog assertions with format strings.
