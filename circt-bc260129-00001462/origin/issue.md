---
type: enhancement
component: ImportVerilog
dialect: Moore
---

# [ImportVerilog] Support concurrent assertions with action blocks

## Description

CIRCT's ImportVerilog component does not support concurrent assertion statements (e.g., `assert property`) that include action blocks (the `else $error(...)` clause). Attempting to use this valid IEEE 1800-2017 construct results in a clear error message indicating the feature is not yet implemented.

## Test Case

**File**: `bug.sv`
```systemverilog
module top(input logic clk);
  logic a, b;
  assert property (@(posedge clk) a |-> b) else $error("msg");
endmodule
```

## Error Message

```
bug.sv:3:3: error: concurrent assertion statements with action blocks are not supported yet
  assert property (@(posedge clk) a |-> b) else $error("msg");
  ^
```

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv
```

## Root Cause

The ImportVerilog component intentionally rejects concurrent assertions with action blocks at this location:

**File**: `lib/Conversion/ImportVerilog/Statements.cpp`
**Function**: `visit(const slang::ast::ConcurrentAssertionStatement &stmt)`
**Lines**: 746-773

```cpp
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

The code explicitly handles assertions **without** action blocks (creates `verif::AssertOp`) but emits "not supported yet" when action blocks are present.

## What IS Supported

Concurrent assertions **without** action blocks work correctly:

```systemverilog
module top(input logic clk);
  logic a, b;
  assert property (@(posedge clk) a |-> b);  // Works
endmodule
```

This is converted to `verif::AssertOp` in the verif dialect.

## Validation

The test case is syntactically valid per IEEE 1800-2017:

| Tool | Status |
|------|--------|
| Slang | ✅ Pass |
| Verilator | ✅ Pass |
| Icarus Verilog | ❌ Limited SVA support (expected) |

## Gap Analysis

- **verif dialect**: `verif::AssertOp` takes `property`, `enable`, and `label` - no message/action block support
- **SV dialect**: `sv.assert.concurrent` has `message` attribute for action blocks
- **Missing**: Conversion path from Slang's `ConcurrentAssertionStatement.ifTrue/ifFalse` to appropriate IR

Additionally, the FIRRTL → ExportVerilog path already supports emitting action blocks for assertions (via intrinsics), showing that the infrastructure exists.

## Suggested Implementation Directions

1. **Option A**: Extend `verif::AssertOp` to support action blocks, then lower to `sv.assert.concurrent`
2. **Option B**: Create `sv.assert.concurrent` directly from ImportVerilog when action blocks are present
3. **Option C**: Convert action block statements to a separate IR construct that gets merged later

## Workaround

Remove the action block (else clause) from concurrent assertions:

```systemverilog
// Instead of:
assert property (@(posedge clk) a |-> b) else $error("msg");

// Use:
assert property (@(posedge clk) a |-> b);
```

## Impact

- **Severity**: Low (graceful error with clear message)
- **Affected Users**: Designers using SystemVerilog Assertions (SVA) with action blocks
- **Standard Compliance**: Partial - IEEE 1800-2017 Section 16.5 defines concurrent assertions with action blocks

## References

- **IEEE 1800-2017**: Section 16.5 - Concurrent assertions
- **Related Issue**: [#7801](https://github.com/llvm/circt/issues/7801) - [ImportVerilog] How to implement SVA in Moore? (broader discussion on SVA support)
- **Related Issue**: [#2486](https://github.com/llvm/circt/issues/2486) - [ExportVerilog] CSE of Assert Action Block Temporaries trips VCS Lint (action blocks in ExportVerilog)

## Labels

`enhancement`, `ImportVerilog`, `Moore`
