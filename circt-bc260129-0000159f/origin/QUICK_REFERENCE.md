# Quick Reference: CIRCT Bug 260129-0000159f

## Crash Summary
- **Error:** `state type must have a known bit width; got '!llhd.ref<i1>'`
- **Tool:** arcilator (CIRCT Arc dialect lowering)
- **Pass:** LowerStatePass
- **Assertion:** StorageUniquerSupport.h:180

## Key Stack Frames
1. **#12** circt::arc::StateType::get(mlir::Type)
2. **#13** ModuleLowering::run() @ LowerState.cpp:219
3. **#14** LowerStatePass::runOnOperation() @ LowerState.cpp:1198

## Root Cause
LLHD reference type `!llhd.ref<i1>` passed to StateType::get() which requires types with known bit width.

## Test Case Trigger
```verilog
module example(input logic clk, inout logic c);  // <- inout becomes !llhd.ref<i1>
  logic [3:0] temp_reg;
  always @(posedge clk) temp_reg <= temp_reg + 1;
  assign c = (a) ? temp_reg[0] : 1'bz;
endmodule
```

## Reproduction Command
```bash
export PATH=/opt/llvm-22/bin:/opt/firtool/bin:$PATH
circt-verilog --ir-hw source.sv | arcilator | opt -O0 | llc -O0 --filetype=obj -o test.o
```

## Generated Artifacts
- `circt_generated.ir` - CIRCT intermediate representation
- `llvm_generated.ir` - LLVM intermediate representation
- `test.o` - Compiled object file
- `reproduce.json` - Structured reproduction data

## Current Status
**NOT REPRODUCED** - The bug appears to be fixed in firtool-1.139.0 with LLVM 22.0.0git

## Files
- `error.txt` - Original crash log with full stack trace
- `source.sv` - Test case source code
- `reproduce.json` - Machine-readable reproduction report
- `crash_signature_analysis.txt` - Detailed signature analysis

