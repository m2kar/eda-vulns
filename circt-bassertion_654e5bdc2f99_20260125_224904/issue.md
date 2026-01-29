<!-- Title: [Arc] Arcilator crashes on modules with inout ports -->
<!-- Labels: Arc, bug, arcilator -->

## Description

Arcilator crashes when processing SystemVerilog modules with `inout` ports. The crash occurs in `LowerStatePass` when it attempts to create `arc::StateType` for an `llhd.ref<i1>` type argument (representing the inout port). The lowering fails because `StateType` does not support `llhd::RefType`.

**Crash Type**: Legalization failure / assertion  
**Dialect**: Arc  
**Failing Pass**: LowerStatePass

## Steps to Reproduce

1. Save the test case below as `bug.sv`
2. Run:
   ```bash
   circt-verilog --ir-hw bug.sv | arcilator --observe-ports
   ```

## Test Case

```systemverilog
module M(inout d);
endmodule
```

## Error Output

```
<stdin>:2:19: error: failed to legalize operation 'arc.state_write'
  hw.module @M(in %d : !llhd.ref<i1>) {
                  ^
<stdin>:2:19: note: see current operation: "arc.state_write"(%5, %7) : (!arc.state<!llhd.ref<i1>>, !llhd.ref<i1>) -> ()
```

## Root Cause Analysis

### Hypothesis (High Confidence)

**Cause**: `LowerState.cpp` lacks filtering or early rejection of `llhd.ref` types when iterating module arguments.

**Evidence**:
- The crash occurs at `LowerState.cpp:219` when iterating all module arguments
- No port direction check exists in the `ModuleLowering::run()` loop
- `computeLLVMBitWidth()` explicitly does not support `llhd::RefType`
- Error message shows `llhd.ref<i1>` being passed to `StateType::get()`
- `ModelOp::verify()` has an inout check but runs too late (after lowering fails)

**Mechanism**:
The `ModuleLowering::run()` function iterates over all module body arguments and creates `RootInputOp` for each. It assumes all arguments have types that can be converted to `StateType`. When an inout port is present, its `llhd.ref` type fails the `StateType` validation since `computeLLVMBitWidth()` doesn't support reference types.

### Processing Path

1. `circt-verilog --ir-hw` converts SystemVerilog to HW IR
2. Inout port becomes an argument with `llhd.ref<T>` type
3. Output is piped to `arcilator`
4. `arcilator` runs `LowerStatePass`
5. `ModuleLowering::run()` iterates all module arguments
6. For the inout port argument: calls `StateType::get(llhd.ref<i1>)`
7. `StateType::verify()` calls `computeLLVMBitWidth(llhd.ref<i1>)`
8. `computeLLVMBitWidth()` returns `std::nullopt` (type not supported)
9. Legalization fails

## Validation

**Cross-Tool Verification**:

| Tool | Status | Notes |
|------|--------|-------|
| Slang | ✅ pass | Build succeeded: 0 errors, 0 warnings |
| Verilator | ✅ pass | No errors or warnings |
| Icarus Verilog | ✅ pass | Successfully compiled |

The test case is valid IEEE 1800 SystemVerilog. All major EDA tools accept this code without issues.

## Environment

- **CIRCT Version**: firtool-1.139.0 (LLVM 22.0.0git)
- **OS**: Linux
- **Architecture**: x86_64

## Suggested Fix Directions

1. **Early Rejection in LowerStatePass** (Recommended):
   Add a check in `ModuleLowering::run()` to skip or error on arguments with `llhd.ref` types:
   ```cpp
   for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
     if (isa<llhd::RefType>(arg.getType())) {
       return moduleOp.emitError("inout ports are not supported in arcilator");
     }
     // ... existing code
   }
   ```

2. **Pipeline Validation Pass**:
   Add a preprocessing pass that validates all modules are arcilator-compatible before lowering begins.

3. **Better Error Message**:
   At minimum, emit a diagnostic before the legalization fails, explaining that inout ports are not supported.

## Related Issues

- #8825: [LLHD] Switch from hw.inout to a custom signal reference type - provides context on `llhd.ref` type system
- #4916: [Arc] LowerState: nested arc.state get pulled in wrong clock tree - same pass, different issue

## Related Files

- `lib/Dialect/Arc/Transforms/LowerState.cpp` - Crash location, needs port direction check
- `lib/Dialect/Arc/ArcTypes.cpp` - `StateType::verify()` and `computeLLVMBitWidth()`
- `lib/Dialect/Arc/ArcOps.cpp` - Existing `ModelOp::verify()` inout check (runs too late)

---
*This issue was generated with assistance from an automated bug reporter.*
