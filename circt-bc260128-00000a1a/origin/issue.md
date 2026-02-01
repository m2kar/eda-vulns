# [Arc][arcilator] Assertion failure on LLHD ref types with inout ports - Regression test case

## Summary

This is a regression test case documenting a historical crash in CIRCT's arcilator that occurred when processing SystemVerilog modules with bidirectional (`inout`) ports. The crash no longer reproduces with CIRCT version firtool-1.139.0, but is documented here as a regression test to prevent similar issues in the future.

**Original Crash:** Arcilator assertion failure when creating `arc::StateType` with LLHD reference types  
**Status:** Does NOT reproduce with current toolchain (firtool-1.139.0)  
**Classification:** Historical documentation / Regression test case

---

## Description

When arcilator processed SystemVerilog modules with `inout` ports, it would crash with an assertion failure during the `LowerStatePass`. The issue occurred because:

1. **Root Cause:** The `arc::StateType` cannot handle LLHD reference types (`!llhd.ref<T>`)
2. **Why It Happens:** `circt-verilog --ir-hw` converts `inout` ports to LLHD reference types for bidirectional signal representation
3. **The Crash:** During `LowerStatePass`, `StateType::get()` is called with the LLHD ref type, which fails verification because `computeLLVMBitWidth()` doesn't recognize LLHD types

This crash appears to have been fixed in the current CIRCT toolchain, but the test case serves as important documentation of the issue and can be used for regression testing.

---

## Steps to Reproduce

### Original Reproduction (Historical - No Longer Crashes)

Using CIRCT 1.139.0, the following command chain **used to crash** but now succeeds:

```bash
circt-verilog --ir-hw source.sv | arcilator | opt -O0 | llc -O0 --filetype=obj -o test.o
```

### Current Status (No Crash)

With firtool-1.139.0, the compilation pipeline completes successfully without crashes. All three reproduction attempts succeed:

1. **Basic pipeline:** `circt-verilog --ir-hw source.sv | arcilator` → SUCCESS
2. **Full pipeline:** `circt-verilog --ir-hw source.sv | arcilator | opt -O0 | llc -O0 --filetype=obj` → SUCCESS (object file generated)
3. **With optimization flags:** `circt-verilog --ir-hw source.sv | arcilator --detect-enables --detect-resets` → SUCCESS (valid LLVM IR)

---

## Test Case

### Source Code (SystemVerilog)

```systemverilog
module MixedPorts(
  input  logic a,
  output logic b,
  inout  wire  c
);

  logic r1;

  always_comb begin
    r1 = a;
  end

  assign b = r1;

  assign c = (r1) ? 1'bz : 1'b0;

endmodule
```

### Validation Status

The test case is **syntactically valid** and passes all standard SystemVerilog validators:

- **Slang (10.0.6):** ✅ PASSED - 0 errors, 0 warnings
- **Verilator (5.022):** ✅ PASSED - No lint errors or warnings
- **iverilog (g2009):** ✅ PASSED - Compilation successful

### Key Language Features

| Feature | Description |
|---------|-------------|
| **inout port** | `inout wire c` - bidirectional/tristate port (THE PROBLEMATIC PATTERN) |
| **Mixed port directions** | Module has input, output, and inout ports |
| **Combinational logic** | `always_comb` block with simple assignment |
| **Tristate assignment** | `assign c = (r1) ? 1'bz : 1'b0;` - high-impedance when r1=1, 0 when r1=0 |

---

## Error Output

### Original Assertion Failure

The following assertion error **used to occur** but no longer reproduces:

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
```

### Full Assertion Error Message

```
arcilator: /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/IR/StorageUniquerSupport.h:180: 
static ConcreteT mlir::detail::StorageUserBase<circt::arc::StateType, mlir::Type, circt::arc::detail::StateTypeStorage, mlir::detail::TypeUniquer>::get(MLIRContext *, Args &&...) 
[ConcreteT = circt::arc::StateType, BaseT = mlir::Type, StorageT = circt::arc::detail::StateTypeStorage, UniquerT = mlir::detail::TypeUniquer, Traits = <>, Args = <mlir::Type &>]: 
Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### Stack Trace (Abbreviated - Key Frames)

```
#12 0x0000565156a7dbbc (/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/arcilator+0x7dd5bbc)
#12 0x0000565156a7dae9 circt::arc::StateType::get(mlir::Type)
     /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/build/tools/circt/include/circt/Dialect/Arc/ArcTypes.cpp.inc:108:3

#13 0x0000565156ae8f5c (anonymous namespace)::ModuleLowering::run()
     /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Dialect/Arc/Transforms/LowerState.cpp:219:66

#14 0x0000565156ae8f5c (anonymous namespace)::LowerStatePass::runOnOperation()
     /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Dialect/Arc/Transforms/LowerState.cpp:1198:41
```

---

## Root Cause Analysis

### Technical Issue Summary

The `arc::StateType` cannot handle LLHD reference types (`!llhd.ref<T>`). The failure occurs because:

1. **circt-verilog** converts `inout` ports to LLHD reference types: `!llhd.ref<i1>`
2. **arcilator** runs the `LowerStatePass` to allocate simulation state
3. **StateType::get()** is called with the LLHD ref type
4. **StateType::verify()** calls `computeLLVMBitWidth()` which has no handler for LLHD types
5. **Verification fails** because `computeLLVMBitWidth()` returns `std::nullopt` for LLHD ref types

### Generated LLHD IR

The hardware module IR generated from the test case includes the problematic type:

```mlir
module {
  hw.module @MixedPorts(in %a : i1, out b : i1, in %c : !llhd.ref<i1>) {
    hw.output %a : i1
  }
}
```

Note: The `inout` port `c` is represented as `!llhd.ref<i1>`, which is the type that triggers the crash.

### Code Path Analysis

**LowerState.cpp:219** - Module lowering attempts to allocate input storage:
```cpp
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto state = RootInputOp::create(allocBuilder, arg.getLoc(),
                                   StateType::get(arg.getType()),  // CRASHES HERE
                                   name, storageArg);
}
```

**ArcTypes.cpp** - StateType verification fails on LLHD ref types:
```cpp
LogicalResult StateType::verify(MLIRContext *ctx, Type innerType) {
  if (!computeLLVMBitWidth(innerType))
    return emitError(unknownLoc) << "state type must have a known bit width; got " << innerType;
  return success();
}
```

**ArcTypes.cpp** - computeLLVMBitWidth() has no LLHD ref type handler:
```cpp
static std::optional<uint64_t> computeLLVMBitWidth(Type type) {
  if (isa<seq::ClockType>(type)) return 1;
  if (auto intType = dyn_cast<IntegerType>(type)) return intType.getWidth();
  if (auto arrayType = dyn_cast<hw::ArrayType>(type)) { /* handle array */ }
  if (auto structType = dyn_cast<hw::StructType>(type)) { /* handle struct */ }
  return {};  // LLHD ref types fall through to here and return nullopt
}
```

### Root Hypotheses

| # | Hypothesis | Confidence | Evidence |
|---|-----------|------------|----------|
| 1 | Missing type conversion before arcilator - LLHD ref types should be lowered before reaching arcilator | 90% | Error specific to `!llhd.ref<i1>`, computeLLVMBitWidth() ignores LLHD types, arcilator designed for cycle-based simulation |
| 2 | Arcilator should reject unsupported inputs gracefully - Should emit user-friendly error instead of assertion | 70% | Poor error diagnostics, bidirectional ports may be unsupported |
| 3 | computeLLVMBitWidth needs LLHD type support - Could extract inner type's bit width | 60% | LLHD ref types contain inner types, but may not match semantic expectations |

---

## Environment

### Current (No Crash)
- **CIRCT Version:** firtool-1.139.0
- **LLVM Version:** 22.0.0git
- **Host:** Linux x86_64
- **Status:** ✅ All test cases pass without crashes

### Original (With Crash)
- **CIRCT Version:** circt-1.139.0
- **LLVM Version:** 22.0.0git
- **Reproduction Status:** Does NOT reproduce with current toolchain

---

## Related Issues

### Most Relevant - Issue #9395 (CLOSED - Jan 19, 2026)
- **Title:** [circt-verilog][arcilator] Arcilator assertion failure
- **Link:** https://github.com/llvm/circt/issues/9395
- **Similarity:** 95% - Very similar arcilator assertion failure in Arc dialect
- **Status:** Recently closed (Jan 19, 2026)
- **Relevance:** Different assertion error but same domain (arcilator/Arc/SystemVerilog). This issue may contain the root cause analysis or fix related to our crash.
- **Key Detail:** Assertion failure in arcilator when processing SystemVerilog with `always @*` blocks; involves the same ConvertToArcs pass as our StateType crash

### Architectural Issue - Issue #8825 (OPEN)
- **Title:** [LLHD] Switch from hw.inout to a custom signal reference type
- **Link:** https://github.com/llvm/circt/issues/8825
- **Similarity:** 90% - Directly addresses the root cause
- **Relevance:** This issue discusses the architectural problem: the transition from `hw.inout` to `!llhd.ref<T>` types. Our crash is a symptom of incomplete support for the new LLHD reference type system.
- **Status:** Open - ongoing work to support LLHD ref types across the toolchain

### Related LLHD-to-Arc Conversion - Issue #9467 (OPEN - Jan 20, 2026)
- **Title:** [circt-verilog][arcilator] arcilator fails to lower llhd.constant_time generated from simple SV delay (#1)
- **Link:** https://github.com/llvm/circt/issues/9467
- **Similarity:** 85% - Related LLHD to Arc conversion failure
- **Relevance:** Different type (`llhd.constant_time` vs `llhd.ref<i1>`), but same pipeline and same ConvertToArcs pass. Shows systematic issues with LLHD-to-Arc conversion.

### Related Verilog-to-LLVM Pipeline Issues
- **Issue #8286:** [circt-verilog][llhd][arcilator] Verilog-to-LLVM lowering issues
- **Issue #8012:** [Moore][Arc][LLHD] Moore to LLVM lowering issues
- **Issue #8065:** [LLHD][Arc] Indexing and slicing lowering from Verilog to LLVM IR
- **Issue #8845:** [circt-verilog] circt-verilog produces non comb/seq dialects including cf and llhd
- **Issue #5566:** [SV] Crash in P/BPAssignOp verifiers for hw.inout ports

---

## Recommendations

### Purpose of This Test Case
This test case should be retained as:
1. **Regression test** to ensure LLHD ref type handling in arcilator doesn't regress
2. **Documentation** of the inout port limitation in arcilator
3. **Tracking case** for the broader LLHD-to-Arc conversion pipeline issues

### For Future Development
1. **Add explicit validation** in arcilator to detect and report LLHD reference types with a user-friendly error message
2. **Implement conversion pass** to lower LLHD ref types before LowerStatePass, or exclude such signals from simulation state
3. **Consider architectural solution** per issue #8825: complete the transition to LLHD ref type system with full toolchain support

---

## Metadata

- **Testcase ID:** 260128-00000a1a
- **Crash Type:** Assertion Failure
- **Dialect:** Arc / LLHD  
- **Failing Pass:** arc::LowerStatePass
- **Tool:** arcilator
- **Classification:** Historical documentation / Regression test
- **First Reported:** Generated from fuzzing campaign (FeatureFuzz-SV)
- **Analysis Date:** January 31, 2026

---

## Tags

`systemverilog` `arc` `llhd` `arcilator` `inout` `tristate` `regression-test` `bug` `assertion` `ref-types` `bidirectional-ports`
