# Validation Report

## Syntax Validation

| Tool | Result | Notes |
|------|--------|-------|
| slang | ✅ **Accepted** | `Build succeeded: 0 errors, 0 warnings` |
| verilator | ✅ **Accepted** | `--lint-only` passes with no errors |
| circt-verilog | ✅ **Accepted** | Produces valid HW IR |

The test case is **syntactically valid SystemVerilog** recognized by all major tools.

## Cross-Tool Validation

### slang Output
```
Top level design units:
    m

Build succeeded: 0 errors, 0 warnings
```

### verilator Output
```
(no output - lint passed)
Exit code: 0
```

### circt-verilog Output
The test case is correctly parsed and produces valid HW IR with:
- `hw.module @m` 
- `seq.firreg` for array state `%a`
- `llhd.combinational` block with loop transformation

## Crash Confirmation

**Crash occurs in:** `arcilator` (not in `circt-verilog`)

**Location:** `InferStateProperties.cpp:211` in `applyEnableTransformation()`

**Root cause:** The pass assumes all state argument types are scalar integers when creating zero constants via `hw::ConstantOp::create`. When processing state derived from unpacked arrays, the type is `!hw.array<NxM>` (not `IntegerType`), causing the internal `cast<mlir::IntegerType>` to fail.

**Crash signature:**
```
error: 'arc.state' op operand type mismatch: operand #2
expected type: '!hw.array<2xi1>'
  actual type: 'i<garbage_value>'
```

The garbage integer value (e.g., `i673700224`) is uninitialized memory being interpreted as a type width, confirming the type safety bug.

## Classification

**Classification: `report`**

**Rationale:**
1. ✅ Test case is valid SystemVerilog (accepted by slang, verilator, circt-verilog)
2. ✅ Crash is reproducible and consistent
3. ✅ Root cause identified: type safety bug in InferStateProperties pass
4. ✅ Not a design limitation but an implementation bug
5. ✅ Other tools handle this pattern correctly

## Triggering Pattern

The minimal trigger requires:
1. **Unpacked array** with registered storage
2. **Loop-based shift pattern** creating enable detection
3. **Registered output** referencing array element

```systemverilog
module m(input clk, output logic out);
  logic a [0:1];                        // ← Unpacked array (trigger #1)
  always_ff @(posedge clk) begin
    out <= a[0];                        // ← Registered output (trigger #3)
    a[0] <= 0;
    for (int i = 1; i < 2; i++)         // ← Loop shift pattern (trigger #2)
      a[i] <= a[i-1];
  end
endmodule
```

## Recommendation

This bug should be reported to the CIRCT project. The fix should add a type check in `applyEnableTransformation()` before calling `hw::ConstantOp::create()` to ensure the type is castable to `IntegerType` or `hw::IntType`, and gracefully handle array types (either by returning failure or implementing proper support).
