# 
#  CIRCT Bug Report
#  Testcase ID: 260129-0000159d
#

<!-- Title: [Arc] Assertion failure when lowering inout ports in sequential logic -->

## Description

Arc dialect's LowerState pass crashes with assertion failure when lowering SystemVerilog modules that contain `inout` ports in sequential logic context. The pass attempts to create a StateType for an `!llhd.ref<i1>` type (LLHD reference type) that results from inout ports, but StateType requires a known bit width type.

**Crash Type**: Assertion
**Dialect**: Arc
**Failing Pass**: LowerState

### Root Cause Summary

Bug triggered by arcilator's LowerState pass attempting to create a StateType for an `!llhd.ref<i1>` type (LLHD reference type) that results from lowering inout ports in SystemVerilog modules. The pass assumes all module arguments are bit-width types but fails to handle LLHD reference types, causing assertion failure: "state type must have a known bit width".

## Steps to Reproduce

1. Save the test case below as `test.sv`
2. Run:
   ```bash
   circt-verilog test.sv --ir-hw 2>&1 | arcilator
   ```

## Test Case

\`\`\`systemverilog
module MixedPorts(input logic clk, input logic a, output logic b, inout logic c);
  logic [3:0] temp_reg;
  
  always @(posedge clk) begin
    for (int i = 0; i < 4; i++) begin
      temp_reg[i] = a;
    end
  end
  
  assign b = temp_reg[0];
  assign c = temp_reg[1];
endmodule
\`\`\`

## Error Output

\`\`\`
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: .../LowerState.cpp:219:66: Assertion \`succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace and instructions to reproduce the bug.
Stack dump:
0.	Program arguments: .../bin/arcilator
 #12 circt::arc::StateType::get(mlir::Type) at ArcTypes.cpp.inc:108:3
 #13 (anonymous namespace)::ModuleLowering::run() at LowerState.cpp:219:66
 #14 (anonymous namespace)::LowerStatePass::runOnOperation() at LowerState.cpp:1198:41
...
Aborted (core dumped)
\`\`\`

## Root Cause Analysis

### Hypothesis (High Confidence)

LowerState pass does not handle LLHD reference types that result from inout ports. When a module contains an inout port, circt-verilog converts it to an `!llhd.ref<T>` type. LowerState pass assumes all module arguments are bit-width types and attempts to create StateType objects for them, but StateType cannot represent reference types, causing assertion failure.

**Evidence**:
- Stack trace shows crash at \`StateType::get(arg.getType())\` where \`arg.getType()\` is \`!llhd.ref<i1>\`
- Test case has an inout port \`c\` that is converted to \`!llhd.ref<i1>\`
- Intermediate MLIR confirms port type is \`!llhd.ref<i1>\`
- Error message explicitly states: "state type must have a known bit width; got '!llhd.ref<i1>'"

**Mechanism**:
1. User writes SystemVerilog with inout port
2. circt-verilog lowers to HW dialect with LLHD ref type for port
3. arcilator's LowerState pass iterates over all module arguments
4. Pass assumes all arguments are bit-width types suitable for state storage
5. \`StateType::get(!llhd.ref<i1>)\` fails verification: reference types don't have bit width
6. Assertion failure triggers abort

## Environment

- **CIRCT Version**: 1.139.0 (original), 22.0.0git (current - bug appears fixed)
- **OS**: Linux
- **Architecture**: x86_64

## Stack Trace

<details>
<summary>Click to expand stack trace</summary>

\`\`\`
#12 circt::arc::StateType::get(mlir::Type) at ArcTypes.cpp.inc:108:3
#13 (anonymous namespace)::ModuleLowering::run() at LowerState.cpp:219:66
#14 (anonymous namespace)::LowerStatePass::runOnOperation() at LowerState.cpp:1198:41
#15 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int)::$_3::operator()() const /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Pass/Pass.cpp:0:19
#16 void llvm::function_ref<void ()>::callback_fn<long>(long) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/ADT/STLFunctionalExtras.h:46:12
#17 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Pass/Pass.cpp:619:17
\`\`\`

</details>

## Related Issues

- #9574: [Arc] Assertion failure when lowering inout ports in sequential logic (HIGHLY SIMILAR - created 2026-02-01)
- #4916: [Arc] LowerState: nested arc.state get pulled in wrong clock tree

## Additional Notes

**⚠️ DUPLICATE NOTICE**: This test case appears to describe the same issue as #9574, which was created on 2026-02-01 (today). The existing issue title "[Arc] Assertion failure when lowering inout ports in sequential logic" matches our problem exactly.

**Recommendation**: This test case is provided as additional verification data for issue #9574 rather than creating a new duplicate issue. If you agree this is the same problem, please add this test case as a comment on #9574.

**Reproduction Status**:
- Bug reproduces in CIRCT 1.139.0 (original version)
- Bug does NOT reproduce in CIRCT 22.0.0git (appears to be fixed)
- Issue #9574 is still OPEN as of 2026-02-01

**Suggested Fix Directions**:
1. Filter out LLHD ref types in LowerState pass to skip creating state storage for them
2. Add conversion pass to transform \`!llhd.ref<T>\` to a type Arc can handle before Arc lowering
3. Add type validation before state allocation to provide better error messages instead of assertion failure

---
*This issue was generated with assistance from an automated bug reporter.*
