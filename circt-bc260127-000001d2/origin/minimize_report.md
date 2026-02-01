# Minimization Report

## Summary
- **Original file**: source.sv (13 lines)
- **Minimized file**: bug.sv (3 lines)
- **Reduction**: 76.9% (10 lines removed)

## Root Cause
The crash is triggered by `sim.fmt.literal` from `$error()` in an immediate assertion inside `always_comb`. The format literal is orphaned (not consumed by a print operation), causing LowerArcToLLVM legalization to fail.

## Key Constructs Preserved
1. `module` declaration with input port
2. `always_comb` procedural block
3. Immediate assertion: `assert(q) else $error("")`

## Reduction Steps

| Iteration | Change | Lines | Result |
|-----------|--------|-------|--------|
| 0 | Original | 13 | Crash |
| 1 | Remove comments, condense | 4 | Crash |
| 2 | Remove clock, always_ff | 3 | Pass (no crash) |
| 3 | Change q to input | 3 | Crash |
| 4 | Simplify assertion condition | 3 | Crash |
| 5 | Empty error message | 3 | Crash |
| 6 | Remove always_comb | 2 | Different error |
| 7 | Remove `logic` keyword | 3 | Crash |

## Removals
- Clock input (`clk`) - not needed
- `always_ff` block - not needed
- `output` direction - changed to `input` (output causes optimization)
- `logic` type keyword - implicit
- Comments - not needed
- Verbose error message - empty string suffices

## Minimal Testcase
```systemverilog
module m(input q);
  always_comb assert(q) else $error("");
endmodule
```

## Reproduction Command
```bash
/opt/firtool/bin/circt-verilog --ir-hw bug.sv 2>&1 | /opt/firtool/bin/arcilator 2>&1
```

## Error Signature
```
<stdin>:3:10: error: failed to legalize operation 'sim.fmt.literal'
    %0 = sim.fmt.literal "Error: "
```
