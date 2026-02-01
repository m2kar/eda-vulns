# Minimization Report

## Testcase ID
260128-00000db6

## Original Testcase
```systemverilog
module array_example(input logic a, input logic b);
  logic [1:0] arr;
  
  assign arr[0] = a;
  
  always_comb arr[1] = b;
endmodule
```
**Size**: 129 characters

## Minimized Testcase
```systemverilog
module m(input a,b);
  logic [1:0] x;
  assign x[0] = a;
  always_comb x[1] = b;
endmodule
```
**Size**: 82 characters

## Reduction Summary
- **Original size**: 129 characters
- **Minimized size**: 82 characters
- **Reduction**: 36.4%

## Minimization Strategy

### Changes Applied
1. **Module name**: `array_example` → `m` (shorter identifier)
2. **Port declarations**: `input logic a, input logic b` → `input a,b` (combined, implicit type)
3. **Array name**: `arr` → `x` (minimal identifier)
4. **Whitespace**: Removed blank lines and trailing spaces

### Preserved Constructs
The following constructs were preserved as they are essential to trigger the bug:
1. **Packed array declaration**: `logic [1:0] x` - creates the 2-bit array
2. **Partial assignment via `assign`**: `assign x[0] = a` - first bit assigned combinationally
3. **Partial assignment via `always_comb`**: `always_comb x[1] = b` - second bit assigned in procedural block

### Why These Constructs Matter
According to root cause analysis:
- The **mixed assignment styles** (continuous `assign` + procedural `always_comb`) to different bits of the same array
- This generates an IR pattern: `extract(concat(...))` 
- The `extractConcatToConcatExtract` canonicalization function handles this pattern
- When `reverseConcatArgs.size() == 1`, it triggers `replaceOpAndCopyNamehint`
- The bug: `eraseOp` is called on an operation that still has uses

## Reproduction Note
⚠️ **Historical Bug**: This crash occurred in CIRCT version 1.139.0 and may be fixed in later versions.
The minimized testcase preserves the essential pattern that triggered the original assertion failure.
