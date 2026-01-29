### Bug report

**CIRCT version**: circt-1.139.0
**Operating system**: Linux
**Description**: Assertion failure "cannot RAUW a value with itself" in Sig2RegPass when processing self-referential continuous assignment combined with procedural assignment

### To Reproduce

```bash
/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw bug.sv
```

Test case:
```systemverilog
module m(input c, output logic q);
  always @(negedge c) q <= 0;
  assign q = q;
endmodule
```

### Expected Behavior

CIRCT should either:
- Emit a clear diagnostic error about the self-referential assignment (`assign q = q`)
- Emit a clear diagnostic about mixed continuous and procedural assignments to the same variable
- Exit gracefully without crashing

Reference: Other tools handle this correctly:
- **Verilator**: Reports `Wire inputs its own output, creating circular logic` and `Blocked and non-blocking assignments to same variable`
- **Slang**: Reports `cannot mix continuous and procedural assignments to variable 'q'`

### Actual Behavior

CIRCT crashes with an assertion failure:
```
circt-verilog: .../mlir/include/mlir/IR/UseDefLists.h:213: void mlir::IRObjectWithUseList<mlir::OpOperand>::replaceAllUsesWith(ValueT &&) [OperandType = mlir::OpOperand, ValueT = mlir::Value &]: Assertion `(!newValue || this != OperandType::getUseList(newValue)) && "cannot RAUW a value with itself"' failed.
```

Stack trace shows crash in Sig2RegPass during signal promotion:
```
#10 (anonymous namespace)::SigPromoter::isPromotable() Sig2RegPass.cpp:207
#11 (anonymous namespace)::Sig2RegPass::runOnOperation() Sig2RegPass.cpp:361
```

### Root Cause Analysis

The test case creates a self-referential continuous assignment (`assign q = q`) combined with an `always` block that also drives `q`. This creates a circular dependency where during signal promotion in Sig2RegPass:

1. The `q` signal is used as both input and output in the continuous assignment
2. When Sig2Reg processes this, it creates an interval where the read value equals the signal being promoted
3. The `createOrFold` operations may fold to produce the original value
4. When the Interval/Offset struct is destroyed, the Value cleanup attempts to replace a value with itself
5. Triggers the assertion: "cannot RAUW a value with itself"

**Key problematic patterns**:
- Self-referential assignment (`assign q = q`)
- Multiple drivers to same signal (always block + continuous assignment)

### Additional Analysis

**Minimization**: Test case reduced from 19 lines to 4 lines (78.9% reduction)

**Validation Results**:
- **Verilator (v5.022)**: Reports semantic error for circular logic and mixed assignments (correct behavior)
- **Slang (v10.0.6)**: Reports error for mixed procedural and continuous assignments (correct behavior)
- **CIRCT**: Should emit diagnostic but crashes instead (compiler robustness bug)

**Duplicate Check**: No existing issues found with this crash pattern in llvm/circt repository. Searched 15 issues with related keywords (Sig2RegPass, RAUW, LLHD, etc.) - no matches for this specific assertion failure.

### Classification

This is a **compiler robustness bug**. While the test case is semantically invalid per IEEE 1800 SystemVerilog (mixing continuous and procedural assignments is not allowed), CIRCT should:
1. Detect the invalid construct during semantic analysis
2. Emit a clear error diagnostic
3. Exit gracefully without crashing

### Suggested Fix Directions

1. **Add self-referential assignment detection**: Before processing intervals, check if a signal reads from itself in a continuous assignment and either emit an error or skip that signal.

2. **Detect multiple drivers early**: Add validation to detect and reject signals with multiple drivers from different source types (combinational vs sequential).

3. **Guard against self-RAUW**: In the `promote()` function, add checks not just for `read != interval.value` but also verify the underlying use-def chains don't form cycles.

### Keywords

Sig2RegPass, LLHD, assertion, RAUW, replaceAllUsesWith, self-referential, multiple-drivers, signal-promotion

### Related Files

- `lib/Dialect/LLHD/Transforms/Sig2RegPass.cpp` - Crash location
- `mlir/include/mlir/IR/UseDefLists.h:213` - RAUW assertion
- `include/circt/Dialect/LLHD/IR/LLHDOps.td` - LLHD operation definitions
