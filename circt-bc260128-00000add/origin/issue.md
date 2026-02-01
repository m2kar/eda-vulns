<!-- 
  CIRCT Bug Report - Historical Issue
  Original Version: circt-1.139.0
  Generated: 2025-01-31
-->

# [Arc] Assertion failure in arcilator LowerStatePass when processing inout ports with llhd.ref type

## Description

Arcilator crashes with an assertion failure when processing a SystemVerilog module containing an `inout` (bidirectional) port. The crash occurs in the `LowerStatePass` when attempting to create an `arc::StateType` for an `!llhd.ref<i1>` type, which is not recognized by `computeLLVMBitWidth()`.

**Root Cause**: Arcilator's `LowerStatePass` does not support LLHD ref types that arise from `inout` ports. The `StateType::verify()` function explicitly rejects any type without a known bit width, and `llhd::RefType` falls into this category.

**Issue Status**: This bug was present in **circt-1.139.0** but appears to be **fixed in the current toolchain** (firtool-1.139.0 + LLVM 22.0.0git).

**Severity**: Medium - The issue prevents simulation of any module with bidirectional ports using arcilator, but has a clear workaround.

**Crash Type**: `assertion`  
**Dialect**: `Arc` (with LLHD types)  
**Failing Pass**: `arc::LowerStatePass`

## Steps to Reproduce

### Historical Reproduction (circt-1.139.0)

1. Save the test case below as `test.sv`
2. Run:
   ```bash
   circt-verilog --ir-hw test.sv | arcilator | opt -O0 | llc -O0 --filetype=obj -o test.o
   ```

### Current Status (firtool-1.139.0 + LLVM 22.0.0git)

The above command now **completes successfully without assertion failure**.

## Test Case

```systemverilog
module m(inout logic x);
endmodule
```

This is a minimal, reproducible test case (2 lines, 77.8% reduction from original). The presence of a single `inout` port is sufficient to trigger the assertion in the original version.

### Key Constructs

- **`inout logic x`**: Bidirectional port, lowered to `!llhd.ref<i1>` in CIRCT IR

### Why This Matters

The test case is syntactically valid (verified by slang 10.0.6, Verilator 5.022, and Icarus Verilog). The failure is **not a user error** but rather a limitation in arcilator's support for bidirectional ports.

## Error Output

### Original Error (circt-1.139.0)

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
Aborted
```

### Stack Trace

```
#12 circt::arc::StateType::get(mlir::Type)
#13 (anonymous namespace)::ModuleLowering::run() at LowerState.cpp:219
#14 (anonymous namespace)::LowerStatePass::runOnOperation() at LowerState.cpp:1198
```

## Root Cause Analysis

### Primary Cause

Arcilator's `arc::LowerStatePass` does not support `!llhd.ref` types that represent bidirectional (`inout`) ports. The issue manifests in two related functions:

1. **`computeLLVMBitWidth()` in `lib/Dialect/Arc/ArcTypes.cpp`**:
   - Only handles: `seq::ClockType`, `IntegerType`, `hw::ArrayType`, `hw::StructType`
   - Returns `nullopt` for `llhd::RefType`

2. **`StateType::verify()` in `lib/Dialect/Arc/ArcTypes.cpp` (line ~78)**:
   ```cpp
   LogicalResult
   StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                     Type innerType) {
     if (!computeLLVMBitWidth(innerType))
       return emitError() << "state type must have a known bit width; got "
                          << innerType;
     return success();
   }
   ```

### Mechanism

During the HW-to-Arc lowering phase, `ModuleLowering::run()` (in `lib/Dialect/Arc/Transforms/LowerState.cpp`, line 219) allocates state storage for module inputs:

```cpp
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto state = RootInputOp::create(allocBuilder, arg.getLoc(),
                                   StateType::get(arg.getType()), ...);
}
```

For modules with `inout` ports:
1. The port's type is `!llhd.ref<i1>` in the IR
2. `StateType::get(arg.getType())` attempts to create a StateType with this LLHD ref type
3. `StateType::verify()` checks `computeLLVMBitWidth()` for the bit width
4. Since `llhd::RefType` is not handled, `computeLLVMBitWidth()` returns `nullopt`
5. The verification fails, triggering the assertion

### Why This Happens

- SystemVerilog `inout` ports are bidirectional (can drive or be driven)
- CIRCT represents these as `!llhd.ref<T>` (reference types)
- Arc dialect, designed for simulation, needs concrete types with known bit widths
- There is no automatic conversion of `llhd.ref<i1>` to a type Arc understands

## Suggested Fixes

### Option 1: Add llhd::RefType Support (if semantically valid)

Extend `computeLLVMBitWidth()` to handle ref types:

```cpp
if (auto refType = dyn_cast<llhd::RefType>(type))
  return computeLLVMBitWidth(refType.getNestedType());
```

**Complexity**: Low  
**Risk**: Requires verification that simulation semantics for bidirectional signals are correct

### Option 2: Early Validation in LowerStatePass (Recommended for immediate fix)

Add a validation check before attempting to create state types:

```cpp
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  if (isa<llhd::RefType>(arg.getType())) {
    return moduleOp.emitError() 
      << "arcilator does not support bidirectional (inout) ports; "
         "see https://github.com/llvm/circt/issues/XXXX";
  }
}
```

**Complexity**: Low  
**Risk**: None - improves error clarity for end users

### Option 3: Update Documentation

Document that arcilator does not support:
- `inout` ports
- Bidirectional signals represented as `llhd.ref` types

## Environment

- **CIRCT Version (Original)**: circt-1.139.0
- **CIRCT Version (Current)**: firtool-1.139.0 + LLVM 22.0.0git
- **OS**: Linux
- **Architecture**: x86_64

## Regression Test Recommendation

This test case should be **added to the regression suite** to prevent re-introduction of this bug:

**File**: `circt/test/Dialect/Arc/arc-simulation-inout-unsupported.mlir` (or similar)

**Purpose**: Ensure arcilator provides clear, user-friendly errors when encountering unsupported `inout` ports.

**Test Content**:
```mlir
// Should produce clear error about inout port not supported
// rather than assertion failure
module m(inout logic x) {
}
```

Or in SystemVerilog directly:
```bash
// RUN: circt-verilog --ir-hw %s -o - | arcilator 2>&1 | FileCheck %s
// CHECK: error: {{arcilator does not support|state type must have a known bit width}}

module m(inout logic x);
endmodule
```

## Related Issues

Based on duplicate analysis (confidence: medium), the following related issues may be of interest:

- **[#8825](https://github.com/llvm/circt/issues/8825)** - `[LLHD]` Switch from hw.inout to a custom signal reference type (OPEN)
- **[#9467](https://github.com/llvm/circt/issues/9467)** - `[circt-verilog][arcilator]` arcilator fails to lower `llhd.constant_time` (OPEN)
- **[#9260](https://github.com/llvm/circt/issues/9260)** - Arcilator crashes in Upload Release Artifacts CI (OPEN)

## Classification

| Aspect | Value |
|--------|-------|
| **Bug Type** | Incomplete feature support |
| **Severity** | Medium |
| **User Impact** | Cannot simulate modules with bidirectional ports using arcilator |
| **Reproducibility** | Deterministic (in circt-1.139.0) |
| **Status** | Fixed in current toolchain |
| **Workaround** | Avoid using `inout` ports in modules processed by arcilator |

## Notes

- This is a **historical bug report** for circt-1.139.0
- The bug appears to have been fixed in the current toolchain version
- The test case is included for **regression testing purposes**
- Recommend verifying which commit fixed this issue in the CIRCT repository

---

*Issue generated with assistance from automated bug reporter. Test case minimized to 2 lines while preserving the crash trigger.*
