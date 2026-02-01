# Root Cause Analysis Report

## Executive Summary
This is a **feature limitation**, not a bug. The CIRCT ImportVerilog component intentionally rejects concurrent assertion statements (e.g., `assert property`) that include action blocks (the `else $error(...)` clause). The feature is not yet implemented in the Slang-to-Moore conversion path.

## Crash Context
- **Tool/Command**: `circt-verilog --ir-hw`
- **Dialect**: Moore (via ImportVerilog)
- **Failing Component**: ImportVerilog (Statements.cpp)
- **Error Type**: Intentional feature limitation error (not a crash/assertion)

## Error Analysis

### Error Message
```
test_391741fc8ca5.sv:10:3: error: concurrent assertion statements with action blocks are not supported yet
  assert property (@(posedge clk) q |-> idx != 0) 
  ^
```

### Error Location in CIRCT Source
**File**: `lib/Conversion/ImportVerilog/Statements.cpp`
**Function**: `visit(const slang::ast::ConcurrentAssertionStatement &stmt)`
**Lines**: 746-773

## Test Case Analysis

### Code Summary
```systemverilog
module top(input logic clk);
  logic [3:0] idx;
  logic q;
  
  always_ff @(posedge clk) begin
    q <= ~q;
    idx <= idx + 1;
  end
  
  assert property (@(posedge clk) q |-> idx != 0) 
    else $error("Assertion failed");  // <-- ACTION BLOCK
    
endmodule
```

### Key Constructs
1. **Concurrent assertion**: `assert property (@(posedge clk) q |-> idx != 0)`
   - Uses clock-based property specification with implication operator `|->`
2. **Action block**: `else $error("Assertion failed")`
   - The "else" clause specifies what to do when the assertion fails
   - This is a SystemVerilog 1800-2017 feature (Section 16.5)

### Problematic Pattern
The combination of:
- Concurrent assertion statement (`assert property`)
- With an action block (`else $error(...)`)

## CIRCT Source Analysis

### Code Context
```cpp
// lib/Conversion/ImportVerilog/Statements.cpp, lines 746-773

// Handle concurrent assertion statements.
LogicalResult visit(const slang::ast::ConcurrentAssertionStatement &stmt) {
  auto loc = context.convertLocation(stmt.sourceRange);
  auto property = context.convertAssertionExpression(stmt.propertySpec, loc);
  if (!property)
    return failure();

  // Handle assertion statements that don't have an action block.
  if (stmt.ifTrue && stmt.ifTrue->as_if<slang::ast::EmptyStatement>()) {
    switch (stmt.assertionKind) {
    case slang::ast::AssertionKind::Assert:
      verif::AssertOp::create(builder, loc, property, Value(), StringAttr{});
      return success();
    case slang::ast::AssertionKind::Assume:
      verif::AssumeOp::create(builder, loc, property, Value(), StringAttr{});
      return success();
    default:
      break;
    }
    mlir::emitError(loc) << "unsupported concurrent assertion kind: "
                         << slang::ast::toString(stmt.assertionKind);
    return failure();
  }

  mlir::emitError(loc)
      << "concurrent assertion statements with action blocks "
         "are not supported yet";
  return failure();
}
```

### Processing Path
1. **Slang parses** the SystemVerilog and creates AST
2. **ImportVerilog** traverses the AST to convert to Moore dialect
3. **ConcurrentAssertionStatement visitor** is called
4. **Check for empty action block**: `stmt.ifTrue->as_if<slang::ast::EmptyStatement>()`
   - If true (no action block): creates `verif::AssertOp` or `verif::AssumeOp`
   - If false (has action block): **emits error** "not supported yet"

### Key Insight
The code explicitly handles the "no action block" case and intentionally rejects the "with action block" case with a clear "not supported yet" message. This is a known limitation, not a bug.

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence) ✓
**Cause**: Feature not yet implemented
**Evidence**:
- The error message explicitly says "not supported yet"
- The code has a clear conditional path that handles assertions without action blocks
- The infrastructure for action blocks exists in SV dialect (sv.assert.concurrent supports `message` attribute)
- ExportVerilog can emit `else $error(...)` for assertions created from FIRRTL

**Mechanism**: 
The ImportVerilog path for concurrent assertions only creates `verif::AssertOp` when there's no action block. The `verif::AssertOp` in the verif dialect doesn't directly support action blocks - the SV dialect's `sv.assert.concurrent` does support messages, but the conversion path from Slang's action block to Moore/verif hasn't been implemented.

## Technical Details

### What IS Supported
1. Concurrent assertions **without** action blocks:
   ```systemverilog
   assert property (@(posedge clk) a |-> b);  // Works
   ```
2. FIRRTL path supports action blocks (via intrinsics):
   ```
   assert__label: assert property (@(posedge clock) _GEN) else $error("message");
   ```

### What is NOT Supported
1. Concurrent assertions **with** action blocks in SystemVerilog import:
   ```systemverilog
   assert property (@(posedge clk) a |-> b) else $error("fail");  // ERROR
   ```

### Gap Analysis
- **verif dialect**: `verif::AssertOp` takes `property`, `enable`, and `label` - no message/action support
- **SV dialect**: `sv.assert.concurrent` has `message` attribute for action blocks
- **Missing**: Conversion from Slang's `ConcurrentAssertionStatement.ifTrue/ifFalse` to appropriate IR

## Suggested Fix Directions
1. **Option A**: Extend `verif::AssertOp` to support action blocks, then lower to `sv.assert.concurrent`
2. **Option B**: Create `sv.assert.concurrent` directly from ImportVerilog when action blocks are present
3. **Option C**: Convert action block statements to a separate IR construct that gets merged later

## Classification

| Attribute | Value |
|-----------|-------|
| **Type** | Feature Limitation |
| **Severity** | Low (graceful error, not a crash) |
| **Is Bug** | No |
| **Workaround** | Remove action block from assertion |

## Keywords for Issue Search
`concurrent assertion` `action block` `assert property` `else $error` `ImportVerilog` `not supported yet`

## Related Files to Investigate
- `lib/Conversion/ImportVerilog/Statements.cpp` - Where the error is emitted
- `include/circt/Dialect/Verif/VerifOps.td` - verif dialect operations
- `include/circt/Dialect/SV/SVVerification.td` - sv.assert.concurrent definition
- `lib/Conversion/ExportVerilog/ExportVerilog.cpp` - How action blocks are emitted
