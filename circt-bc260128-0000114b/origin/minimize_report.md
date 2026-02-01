# Minimize Report

## Summary
- **Original file**: source.sv (39 lines)
- **Minimized file**: bug.sv (18 lines)
- **Reduction**: 53.8%

## Key Constructs Preserved
- `struct packed` - packed struct type definition
- `unpacked array` - array of packed structs `pkt_t arr [0:1]`
- `always_ff` - sequential logic block
- `for-loop` - loop construct within always_ff

## Constructs Removed
- Output port `result` - not needed for crash
- `always_comb` block - not needed for crash
- Wider data types (i32 → i8) - simpler types sufficient
- Larger array size (4 → 2 elements) - smaller size sufficient
- Extra struct fields (header, payload → a) - single field sufficient

## Trigger Conditions
The crash requires ALL of the following:
1. **Packed struct** type used in unpacked array
2. **Intermediate register** (`d`) to create enable pattern
3. **For-loop** that generates mux-based enable detection
4. The loop updates struct array elements conditionally

## Crash Analysis
- **Pass**: `arc-infer-state-properties`
- **Function**: `applyEnableTransformation()` at line 211
- **Error**: `cast<IntegerType>` fails on packed struct type
- **Root cause**: `hw::ConstantOp::create()` only supports IntegerType/IntType, but receives struct type from state element

## Reproduction Command
```bash
circt-verilog --ir-hw bug.sv | arcilator
```
