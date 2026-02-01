# CIRCT Crash Report

## Summary
`circt-verilog` asserts in LLHD Sig2RegPass with **"cannot RAUW a value with itself"** when compiling a small SystemVerilog module containing a simple combinational self-loop.

## Crash Description
- **Crash type**: assertion
- **Assertion**: `cannot RAUW a value with itself`
- **Location**: `mlir/IR/UseDefLists.h:213`
- **Pass**: `lib/Dialect/LLHD/Transforms/Sig2RegPass.cpp` (`SigPromoter::promote`)

## Reproduction Steps
```bash
export PATH=/opt/llvm-22/bin:$PATH
/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw bug.sv
```

## Error Log (excerpt)
```
circt-verilog: /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/IR/UseDefLists.h:213: void mlir::IRObjectWithUseList<mlir::OpOperand>::replaceAllUsesWith(ValueT &&) [OperandType = mlir::OpOperand, ValueT = mlir::Value &]: Assertion `(!newValue || this != OperandType::getUseList(newValue)) && "cannot RAUW a value with itself"' failed.
PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
Stack dump:
0.	Program arguments: /edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw source.sv
#12 (anonymous namespace)::SigPromoter::isPromotable() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Dialect/LLHD/Transforms/Sig2RegPass.cpp:207:11
#13 (anonymous namespace)::Sig2RegPass::runOnOperation() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Dialect/LLHD/Transforms/Sig2RegPass.cpp:361:58
...
```

## Minimal Testcase (bug.sv)
```systemverilog
module example_module(output logic [3:0] out);
  logic [3:0] internal_wire;
  
  assign internal_wire[3:0] = out[3:0];
  assign out = internal_wire;
endmodule
```

## Root Cause Analysis (summary)
The LLHD Sig2Reg promotion replaces reads via `replaceAllUsesWith`. For this self-looping signal, the
replacement value aliases the same use list as the original, triggering the MLIR self-RAUW assertion.

## Related Files/Paths
- Crash log: `origin/error.txt`
- Reproduction log: `origin/reproduce.log`
- Minimized testcase: `origin/bug.sv`
- Root cause report: `origin/root_cause.md`
- Structured analysis: `origin/analysis.json`
- Validation report: `origin/validation.md`
- Duplicate check: `origin/duplicates.md`

## Notes
- `/opt/firtool/bin/circt-verilog` hangs on this input; reproduction succeeds with the FeatureFuzz-SV build.
