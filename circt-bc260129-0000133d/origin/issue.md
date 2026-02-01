<!-- Title: [Sim] Missing legalization pattern for sim.fmt.literal in ArcToLLVM pass -->

## Description

CIRCT's arcilator tool fails to legalize SystemVerilog code containing assertions with `$error` system tasks in combinational always blocks. The error occurs in the ArcToLLVM pass when attempting to lower `sim.fmt.literal` operations that are generated from `$error` calls but appear outside of a procedural print context.

The issue is that `sim.fmt.literal` operations lack a proper conversion pattern in the ArcToLLVM lowering pass when they are not consumed by `sim.proc.print`. This causes a legalization failure that prevents valid SystemVerilog assertions from being compiled and simulated.

**Crash Type**: Legalization failure  
**Dialect**: sim  
**Failing Pass**: ArcToLLVM

## Steps to Reproduce

1. Save the following SystemVerilog code to `test.sv`:
```systemverilog
module m();
  always @(*) assert (0) else $error("");
endmodule
```

2. Run the CIRCT compilation pipeline:
```bash
circt-verilog --ir-hw test.sv 2>&1 | arcilator 2>&1
```

## Test Case

```systemverilog
module m();
  always @(*) assert (0) else $error("");
endmodule
```

## Error Output

```
<stdin>:3:10: error: failed to legalize operation 'sim.fmt.literal'
    %0 = sim.fmt.literal "Error: "
         ^
<stdin>:3:10: note: see current operation: %0 = "sim.fmt.literal"() <{literal = "Error: "}> : () -> !sim.fstring
```

## Root Cause Analysis

### Summary

The `sim.fmt.literal` operation is a format string literal operation from CIRCT's Sim dialect. When a SystemVerilog `$error()` system task appears within an assertion in a combinational always block, the ImportVerilog pass converts it to `sim.fmt.literal` operations. However, these format literal operations lack a proper lowering pattern in the ArcToLLVM conversion pass.

The root cause occurs in two phases:

1. **ImportVerilog Phase**: Correctly generates `sim.fmt.literal "Error: "` from the `$error("")` call
2. **ArcToLLVM Phase**: Fails to find a conversion pattern for the standalone `sim.fmt.literal` operation

### Technical Root Cause

- **Location**: `lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp`
- **Trigger Construct**: Assertion with `$error` in combinational always block
- **Root Cause**: Missing conversion pattern for standalone `sim.fmt.literal` operations in the ArcToLLVM pass
- **Why It Happens**: 
  - The SimToSV pass does not handle format string operations outside of print contexts
  - The ArcToLLVM pass expects format operations to be consumed by `SimPrintFormattedProcOp`
  - When `sim.fmt.literal` appears orphaned (not part of a print hierarchy), no conversion pattern matches
  - The operation is marked as illegal by the dialect conversion framework, but no legalization pattern is provided

### Affected Passes

- ImportVerilog: Generates the problematic `sim.fmt.literal` operation
- ArcToLLVM: Fails to lower it to LLVM

## Validation

The test case has been validated as correct SystemVerilog:
- **slang 10.0.6**: ✅ Passes (`--parse-only`)
- **verilator 5.022**: ✅ Passes (`--lint-only --sv`)

This confirms the issue is a CIRCT/arcilator legalization gap, not invalid input syntax.

## Environment

- **CIRCT Version**: circt-1.139.0
- **OS**: Linux 6.2.0
- **Architecture**: x86_64

## Related Issues

Based on repository search, no exact duplicates were found. However, related patterns include:

1. **Issue #8012** - [Moore][Arc][LLHD] Moore to LLVM lowering issues
   - Similarity: 4.8/10 (also missing legalization patterns in arcilator)
   
2. **Issue #9467** - [circt-verilog][arcilator] arcilator fails to lower llhd.constant_time
   - Similarity: 2.8/10 (similar missing conversion pattern but for different operation)

The current issue is unique in specifically targeting `sim.fmt.literal` in assertion contexts.

## Suggested Fix Directions

1. **Option A**: Add a legalization pattern to lower standalone `sim.fmt.literal` to a no-op or constant in ArcToLLVM when not used by a print operation

2. **Option B**: In ImportVerilog, avoid generating orphaned `sim.fmt.literal` for assertions, or wrap them in appropriate print operations

3. **Option C**: Add dead code elimination (DCE) before ArcToLLVM to remove unused format string operations

4. **Option D**: Update the conversion target in ArcToLLVM to mark `sim.fmt.literal` as legal when orphaned, allowing it to be eliminated later

---

*This issue was generated with assistance from an automated bug reporter.*
