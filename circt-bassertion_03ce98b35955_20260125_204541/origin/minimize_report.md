# Minimization Report

## Summary
- **Original file**: source.sv (14 lines)
- **Minimized file**: bug.sv (3 lines)
- **Reduction**: 78.6%
- **Crash preserved**: Yes

## Preservation Analysis

### Key Constructs Preserved
Based on `analysis.json`, the following constructs were kept:
- `packed union` - the core trigger for the crash
- `module port with user-defined type` - union type used as port type

### Removed Elements
- Module `Top` (4 lines) - not needed to trigger crash
- Module `Sub` body content (1 line) - assign statement not needed
- Union member `b` - single member sufficient
- Type aliases with full bit widths - minimized to `logic a`

## Verification

### Original Assertion Pattern
```
SVModuleOpConversion::matchAndRewrite -> MooreToCorePass crash
```

### Final Assertion Pattern
```
SVModuleOpConversion::matchAndRewrite -> MooreToCorePass crash
```

**Match**: ✅ Same crash path confirmed

## Reproduction Command

```bash
/opt/firtool/bin/circt-verilog --ir-hw bug.sv
```

## Minimized Test Case

```systemverilog
typedef union packed { logic a; } my_union;
module Sub(input my_union x);
endmodule
```

## Root Cause
MooreToCore type conversion lacks handler for `moore::UnionType`, causing null type propagation and assertion failure when packed unions are used as module port types.

## Notes
- The crash is triggered by any packed union used as a module port type
- Both input and output ports with union types cause the crash
- The union can have any number of members (even just one)
- Typedef is optional - inline union definition also crashes
