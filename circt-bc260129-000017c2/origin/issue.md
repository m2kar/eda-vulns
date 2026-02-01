# Fixed Bug Report: arcilator assertion failure with inout ports

## Summary

This issue documents a bug that existed in CIRCT 1.139.0 and has been **fixed** in the current toolchain (firtool-1.139.0 with LLVM 22.0.0git).

**Status**: ✅ **ALREADY FIXED** - No action required

## Original Crash Details

### Test Case
```systemverilog
module MixedPorts(
  input  logic clk,
  input  logic a,
  output logic b,
  inout  logic c
);

  logic sig;
  logic c_drive;

  // Inout port bidirectional behavior
  assign c = c_drive;

  // Shared source signal assignment from inout driver
  assign sig = c_drive;

  // Same source signal driving multiple destinations
  always_ff @(posedge clk) begin
    a <= sig;
    b <= sig;
  end

endmodule
```

### Error Message
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: .../LowerState.cpp:219: Assertion `succeeded( ConcreteT::verifyInvariants(...))' failed.
```

### Command
```bash
circt-verilog --ir-hw source.sv | arcilator | opt -O0 | llc -O0 --filetype=obj -o test.o
```

### Stack Trace
```
#0  StateType::get(mlir::Type) at ArcTypes.cpp.inc:108
#1  ModuleLowering::run() at LowerState.cpp:219
#2  LowerStatePass::runOnOperation() at LowerState.cpp:1198
```

## Root Cause Analysis

### Technical Details

The crash occurred in `arcilator`'s `LowerState` pass when processing SystemVerilog modules with `inout` ports:

1. **Type Conversion**: When `circt-verilog --ir-hw` processes modules with `inout` ports, it represents bidirectional ports using `!llhd.ref<T>` types

2. **LowerState Pass Bug**: The `LowerState` pass at line 219 iterates over all module arguments and attempts to create `StateType` for each input:
   ```cpp
   auto state = RootInputOp::create(allocBuilder, arg.getLoc(),
                                    StateType::get(arg.getType()), name, storageArg);
   ```

3. **Type Verification Failure**: `StateType::verify()` calls `computeLLVMBitWidth()` which only supports:
   - `seq::ClockType`
   - `IntegerType`
   - `hw::ArrayType`
   - `hw::StructType`

4. **Missing Type Handling**: `llhd::RefType` is NOT among the supported types, so `computeLLVMBitWidth()` returns `{}` (empty optional), causing the verification assertion to fail

### Triggering Pattern
Any SystemVerilog module containing `inout` port declarations:
```systemverilog
inout logic c;  // This triggers the bug in old toolchain
```

### Likely Fix Applied
The fix was likely one of the following:
1. Filter out `llhd::RefType` arguments before creating `RootInputOp`
2. Extend `computeLLVMBitWidth()` to handle `RefType` (e.g., as 64-bit pointer)
3. Emit a proper user-facing error diagnostic instead of assertion failure

## Verification

| Toolchain | Result | Date |
|-----------|--------|------|
| `circt-1.139.0` | ❌ Assertion failure | Original |
| `firtool-1.139.0` (LLVM 22.0.0git) | ✅ Compiles successfully | Current |

The test case now compiles without errors in the current toolchain.

## Impact

This bug affected:
- **Severity**: Medium - Crash during compilation
- **Scope**: Any SystemVerilog code with `inout` ports processed through `arcilator`
- **Component**: `circt::arc::LowerState` pass

## Recommendation

✅ **No action required** - This bug has already been fixed.

This report serves as:
1. Documentation of a resolved issue
2. A regression test case to prevent future breakage
3. Reference for similar type handling issues in other passes

## Files Generated

- `source.sv` - Original test case
- `error.txt` - Original crash log (circt-1.139.0)
- `metadata.json` - Reproduction metadata (confirms crash is fixed)
- `analysis.json` - Detailed root cause analysis
- `root_cause.md` - Complete technical analysis
- `reproduce.log` - Successful compilation output (current toolchain)

---

**Report Generated**: 2026-02-01
**Bug Status**: Resolved / Fixed in current toolchain
