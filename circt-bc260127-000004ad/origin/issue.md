# [Verification] arcilator crash with inout port and tristate assignment (Issue #260127-000004ad)

**Test Case ID**: 260127-000004ad
**Original Report Date**: 2025-01-27
**Verification Date**: 2026-01-31
**Status**: ✅ **VERIFIED FIXED**

---

## Summary

This issue documents a historical crash in arcilator when processing SystemVerilog designs with `inout` ports and tristate assignments. **The bug has been verified as FIXED in the current toolchain** (CIRCT firtool-1.139.0 with LLVM 22.0.0git).

This report serves as:
- Documentation of the original bug
- Root cause analysis for historical reference
- Verification evidence that the fix is effective

---

## Original Bug Report

### Crash Information

| Field | Value |
|-------|-------|
| **Tool** | arcilator |
| **Command** | `circt-verilog --ir-hw source.sv \| arcilator` |
| **Dialect** | Arc/LLHD |
| **Failing Pass** | LowerStatePass (Arc Transforms) |
| **Crash Type** | Assertion failure |
| **Crash Location** | `LowerState.cpp:219` → `StateType::get()` |

### Error Message

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: .../mlir/include/mlir/IR/StorageUniquerSupport.h:180:
Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### Stack Trace (Original)

```
#12 circt::arc::StateType::get(mlir::Type) ArcTypes.cpp.inc:108
#13 (anonymous namespace)::ModuleLowering::run() LowerState.cpp:219
#14 (anonymous namespace)::LowerStatePass::runOnOperation() LowerState.cpp:1198
```

---

## Test Case

### Source Code (source.sv)

```systemverilog
module my_module (
  input  logic enable,
  inout  logic io_sig
);

  logic out_val;

  assign out_val = enable;
  assign io_sig = (out_val) ? 1'b1 : 1'bz;

endmodule
```

### Key Language Constructs

| Construct | Role in Bug |
|-----------|-------------|
| `inout logic io_sig` | Bidirectional port converted to `llhd::RefType` |
| `assign io_sig = ... ? 1'b1 : 1'bz` | Tristate conditional assignment |
| `1'bz` high-impedance | Tristate logic triggered LLHD dialect usage |

---

## Root Cause Analysis

### Mechanism

1. **circt-verilog --ir-hw** compiles SystemVerilog to HW/LLHD mixed IR
2. **inout port** is represented as `llhd::RefType<i1>` type
3. **arcilator** runs LowerState pass to allocate storage for module outputs
4. **ModuleLowering::getAllocatedState()** calls `StateType::get(result.getType())`
5. `result.getType()` is `llhd::RefType<i1>`
6. **StateType::verify()** calls `computeLLVMBitWidth(llhd::RefType<i1>)`
7. `computeLLVMBitWidth()` **doesn't recognize llhd::RefType**, returns `std::nullopt`
8. Verification fails, triggers assertion

### Root Cause

**High Confidence**: `arc::StateType`'s verification logic (`computeLLVMBitWidth`) only supports a limited type set (`seq::ClockType`, `IntegerType`, `hw::ArrayType`, `hw::StructType`), excluding `llhd::RefType`.

**Code Location**: `lib/Dialect/Arc/ArcTypes.cpp` (~lines 75-87)

```cpp
static std::optional<uint64_t> computeLLVMBitWidth(Type type) {
  if (isa<seq::ClockType>(type))
    return 1;

  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();

  if (auto arrayType = dyn_cast<hw::ArrayType>(type)) { ... }
  if (auto structType = dyn_cast<hw::StructType>(type)) { ... }

  // We don't know anything about any other types.
  return {};  // <-- llhd::RefType would reach here
}
```

### Impact

- **Severity**: High - completely blocks simulation of designs with inout/tristate logic
- **Scope**: Any SystemVerilog design using inout ports or tristate logic through arcilator

---

## Verification Results

### Current Toolchain

| Tool | Version |
|------|---------|
| CIRCT | firtool-1.139.0 |
| LLVM | 22.0.0git |
| Date | 2026-01-31 |

### Reproduction Command

```bash
export PATH=/opt/llvm-22/bin:$PATH
circt-verilog --ir-hw source.sv | arcilator | opt -O0 | llc -O0 --filetype=obj -o test.o
```

### Verification Steps

| Step | Command | Result |
|------|---------|--------|
| 1. circt-verilog --ir-hw | `circt-verilog --ir-hw source.sv` | ✅ Success - Generates HW IR with `!llhd.ref<i1>` |
| 2. arcilator | `arcilator` | ✅ Success - No crash, converts to LLVM IR |
| 3. opt -O0 | `opt -O0` | ✅ Success - Optimizes LLVM IR |
| 4. llc -O0 | `llc -O0 --filetype=obj` | ✅ Success - Generates object file |

### Verification Summary

| Metric | Value |
|--------|-------|
| **Reproduced** | ❌ No |
| **Exit Code** | 0 (Success) |
| **Signature** | No assertion failure |
| **LLHD Type Present** | Yes (`!llhd.ref<i1>` correctly handled) |

### Conclusion

✅ **The original crash no longer occurs.** The current toolchain correctly handles `llhd::RefType` values generated from inout ports, either by:
1. Adding `llhd::RefType` support to `computeLLVMBitWidth()`, or
2. Converting/normalizing LLHD types before the LowerState pass, or
3. Other architectural improvements

---

## Files Referenced

- `source.sv` - Original test case
- `error.txt` - Original crash log
- `reproduce.log` - Full verification execution log
- `root_cause.md` - Detailed root cause analysis
- `analysis.json` - Structured analysis data
- `metadata.json` - Reproduction metadata

---

## Related Components

| Component | Path |
|-----------|------|
| State Type Definition | `include/circt/Dialect/Arc/ArcTypes.td` |
| State Type Verification | `lib/Dialect/Arc/ArcTypes.cpp` |
| LowerState Pass | `lib/Dialect/Arc/Transforms/LowerState.cpp` |
| Inout Conversion | `lib/Conversion/MooreToCore/MooreToCore.cpp` |

---

## Keywords

`arcilator` `llhd.ref` `StateType` `bit width` `inout` `tristate` `LowerState` `computeLLVMBitWidth`

---

## Note to Developers

This verification report confirms that the issue reported on 2025-01-27 (Test Case ID: 260127-000004ad) has been resolved in the current toolchain. The root cause analysis provided above documents the original failure mechanism for historical reference and regression prevention.

**Recommendation**: Consider adding this test case to the regression test suite to prevent future regressions.
