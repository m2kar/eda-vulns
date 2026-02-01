# Minimization Report

## Summary

| Metric | Value |
|--------|-------|
| Original lines | 8 |
| Minimized lines | 5 |
| Reduction | 37.5% |
| Status | **Minimized** |

## Original Test Case

```systemverilog
module test_module;
  logic [7:0] arr = 8'h0;
  int idx = 0;
  
  always_comb begin
    arr[idx] = 1'b1;
  end
endmodule
```

## Minimized Test Case

```systemverilog
module m;
  logic [3:0] a;
  int i;
  always_comb a[i] = 1'b1;
endmodule
```

## Minimization Steps

### Step 1: Module Name
- **Original:** `test_module`
- **Minimized:** `m`
- **Rationale:** Module name is irrelevant to the bug

### Step 2: Array Size
- **Original:** `[7:0]` (8 bits)
- **Minimized:** `[3:0]` (4 bits)
- **Rationale:** Bug is triggered by ExtractOp/ConcatOp pattern from dynamic indexing. 4 bits is sufficient to create the same IR pattern.

### Step 3: Variable Initialization
- **Original:** `= 8'h0` and `= 0`
- **Minimized:** Removed
- **Rationale:** Initializations don't affect the canonicalization path that triggers the crash

### Step 4: always_comb Block
- **Original:** Multi-line with `begin`/`end`
- **Minimized:** Single-line
- **Rationale:** Syntactic sugar only; same semantic meaning

### Step 5: Variable Names
- **Original:** `arr`, `idx`
- **Minimized:** `a`, `i`
- **Rationale:** Names don't affect compilation behavior

## Essential Elements Preserved

1. **Dynamic Index Variable** (`int i`): Required to trigger dynamic array access pattern
2. **Logic Array** (`logic [n:0] a`): Target of dynamic indexed assignment
3. **always_comb Block**: Triggers combinational logic path and canonicalization
4. **Indexed Assignment** (`a[i] = 1'b1`): Core construct that generates ExtractOp with ConcatOp

## Rejected Minimizations

| Attempt | Rejected Because |
|---------|-----------------|
| `[1:0]` array | Still valid, but 4 bits provides clearer IR pattern |
| Static index `a[0]` | Would not trigger dynamic indexing code path |
| Remove always_comb | Required for the combinational canonicalization path |

## Conclusion

The minimized test case reduces the original 8 lines to 5 lines (37.5% reduction) while preserving all essential elements needed to trigger the `extractConcatToConcatExtract` assertion failure in CIRCT v1.139.0.
