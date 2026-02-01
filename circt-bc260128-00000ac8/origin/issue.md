# [Arc] Assertion failure when processing packed struct in sequential logic with enable pattern

## Bug Description

Arcilator crashes with an assertion failure when processing SystemVerilog code containing a **packed struct array in sequential logic** with a shift register pattern. The crash occurs in the `InferStateProperties` pass during the `applyEnableTransformation` function when attempting to create a hardware constant operator.

The root cause is a missing type check: the code assumes all state elements can be represented as integer types when optimizing enable patterns, but packed structs are aggregate types that cannot be used with `hw::ConstantOp::create` in this context.

## Minimized Testcase

```systemverilog
// Minimal testcase: packed struct array shift with for-loop triggers
// hw::ConstantOp type assertion failure in InferStateProperties pass
module test(
  input logic clk,
  output logic o
);
  struct packed { logic d; } s[2];

  always @(posedge clk) begin
    for (int i = 1; i < 2; i++)
      s[i] <= s[i-1];
    o <= s[1].d;
  end
endmodule
```

## Reproduction Steps

1. Save the testcase to `bug.sv`
2. Generate HW dialect IR:
   ```bash
   circt-verilog --ir-hw bug.sv > hw.mlir
   ```
3. Process with arcilator:
   ```bash
   arcilator hw.mlir
   ```
   Or as a pipeline:
   ```bash
   circt-verilog --ir-hw bug.sv | arcilator
   ```

## Expected Behavior

The code should either:
1. Successfully compile and generate correct arcilator output, or
2. Report a meaningful error about unsupported struct types in enable optimization

## Actual Behavior

Arcilator crashes with an assertion failure:

```
arcilator: /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/Casting.h:566: 
decltype(auto) llvm::cast(From &) [To = mlir::IntegerType, From = mlir::Type]: 
Assertion `isa<To>(Val) && "cast<Ty>() argument of incompatible type!"' failed.
```

## Stack Trace

```
 #11 0x0000555cedf74f42 (/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/arcilator+0x87d5f42)
 #12 0x0000555cedf84a5f circt::hw::ConstantOp::create(mlir::OpBuilder&, mlir::Location, mlir::Type, long)
           /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/build/tools/circt/include/circt/Dialect/HW/HW.cpp.inc:2591:55
 #13 0x0000555ced5cb382 mlir::Operation::getOpResultImpl(unsigned int)
           /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/IR/Operation.h:1010:25
 ...
 #17 0x0000555ced5cb382 applyEnableTransformation(circt::arc::DefineOp, circt::arc::StateOp, llvm::ArrayRef<(anonymous namespace)::EnableInfo>)
           /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Dialect/Arc/Transforms/InferStateProperties.cpp:211:55
 #18 InferStatePropertiesPass::runOnStateOp(...)
           /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Dialect/Arc/Transforms/InferStateProperties.cpp:454
```

## Root Cause

### Type Assumption Violation in Enable Transformation

The crash occurs in `lib/Dialect/Arc/Transforms/InferStateProperties.cpp` at line 211 in the `applyEnableTransformation` function:

```cpp
inputs[enableInfos[i].selfArg.getArgNumber()] = hw::ConstantOp::create(
    builder, stateOp.getLoc(), enableInfos[i].selfArg.getType(), 0);
```

**The Problem**: 
- The code assumes `selfArg.getType()` is an `mlir::IntegerType`
- `hw::ConstantOp::create` with a `long` value (0) only supports integer types
- When the state element is a packed struct (non-integer aggregate type), the internal cast to `IntegerType` fails with an assertion

### Why This Triggers

The `InferStateProperties` pass attempts to optimize state operations by detecting reset/enable patterns. For enable patterns detected via `getIfMuxBasedEnable`, it replaces the self-looping input with a constant zero value. However:

1. The test pattern creates an arc state operation from `s[i] <= s[i-1]` 
2. The packed struct is represented as a non-integer type (`hw::StructType`)
3. The pass detects this as an enable pattern
4. It attempts to create a zero constant for all types indiscriminately
5. The `hw::ConstantOp::create` call fails because it cannot handle struct types with integer values

### Affected Components

| Component | Impact |
|-----------|--------|
| `lib/Dialect/Arc/Transforms/InferStateProperties.cpp:211` | Direct crash location - missing type check |
| `hw::ConstantOp::create` | Insufficient type validation |
| Arc/Arcilator pipeline | Fails on any struct-based state with enable patterns |

## Additional Context

- **Testcase ID**: 260128-00000ac8
- **Original Code**: 27 lines
- **Minimized Code**: 14 lines
- **Reduction**: 48% lines removed
- **Dialect**: Arc (Arcilator)
- **Pass**: InferStateProperties
- **Crash Type**: Assertion failure in `llvm::cast<mlir::IntegerType>`
- **Confidence**: HIGH (90%)

### Validation Summary

| Tool | Result | Notes |
|------|--------|-------|
| Slang v10.0.6 | ✅ PASS | Valid SystemVerilog syntax |
| Verilator v5.022 | ✅ PASS | Passes linting |
| Arcilator v1.139.0 | ❌ CRASH | Assertion in InferStateProperties |

### Duplicate Check

Reviewed existing CIRCT issues:
- **#9260** "Arcilator crashes in Upload Release Artifacts CI": Similarity score 12.0/20
- **#6373** "[Arc] Support hw.wires of aggregate types": Related but distinct issue
- **Conclusion**: This is a **distinct issue** - crashes on specific struct array shift pattern in InferStateProperties, not a duplicate

### Suggested Fixes

**Option 1: Type Guard (Recommended)**
Add type check in `InferStateProperties.cpp:211` before creating constant:
```cpp
if (enableInfos[i].selfArg.getType().isa<mlir::IntegerType>()) {
  inputs[enableInfos[i].selfArg.getArgNumber()] = hw::ConstantOp::create(
      builder, stateOp.getLoc(), enableInfos[i].selfArg.getType(), 0);
} else {
  // Skip optimization for non-integer types
  return failure();
}
```

**Option 2: Early Pattern Rejection**
Modify pattern detection to reject non-integer types early:
```cpp
if (!selfArg.getType().isa<mlir::IntegerType>())
  return {};  // Skip enable optimization for aggregate types
```

**Option 3: Aggregate Support**
Use `hw::AggregateConstantOp` for struct types instead of `hw::ConstantOp`.

### Impact Assessment

- **Severity**: Medium-High (compiler crash on valid SystemVerilog)
- **Frequency**: Any module using packed structs/arrays with sequential enable-like patterns
- **User Workaround**: None at the SystemVerilog level; requires CIRCT code fix
- **Affected Users**: Any user with packed struct state elements

## Related Issues

This is a type-safety issue in Arc dialect's optimization passes:
- Related to #6373: Support for hw.wires of aggregate types
- Potential similar issues in other passes assuming IntegerType for state values
