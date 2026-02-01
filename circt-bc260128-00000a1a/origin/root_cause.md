# Root Cause Analysis Report

## Crash Summary

| Field | Value |
|-------|-------|
| **Crash Type** | Assertion Failure |
| **Testcase ID** | 260128-00000a1a |
| **Tool** | arcilator |
| **Failing Pass** | arc::LowerStatePass |
| **Dialect** | Arc / LLHD |

## Error Message

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
```

**Assertion:**
```
Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

## Stack Trace Analysis

| # | Function | File | Line |
|---|----------|------|------|
| 12 | `circt::arc::StateType::get(mlir::Type)` | ArcTypes.cpp.inc | 108 |
| 13 | `ModuleLowering::run()` | LowerState.cpp | 219 |
| 14 | `LowerStatePass::runOnOperation()` | LowerState.cpp | 1198 |

**Crash Location:** `lib/Dialect/Arc/ArcTypes.cpp` in `StateType::verify()`

## Test Case Analysis

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

### Language Features Used
- **Language:** SystemVerilog
- **Key Constructs:**
  - `inout wire c` - **bidirectional/tristate port** (THE PROBLEMATIC PATTERN)
  - `always_comb` block
  - Conditional tristate assignment (`? 1'bz : 1'b0`)
  - Mixed port directions (input, output, inout)

### Problematic Pattern
The **`inout wire c`** port declaration is the trigger for this crash. When SystemVerilog is lowered through the CIRCT toolchain:

1. `circt-verilog --ir-hw` converts the SystemVerilog to HW dialect IR
2. The `inout` port is represented using LLHD dialect's reference type: `!llhd.ref<i1>`
3. When `arcilator` runs, the `LowerStatePass` attempts to allocate storage for values
4. `StateType::get()` is called with the LLHD ref type
5. `StateType::verify()` fails because `!llhd.ref<i1>` cannot compute a known bit width

## Root Cause

### Primary Cause
The `arc::StateType` cannot handle LLHD reference types (`!llhd.ref<T>`). The `computeLLVMBitWidth()` function in `ArcTypes.cpp` only handles:
- `seq::ClockType`
- `IntegerType`
- `hw::ArrayType`
- `hw::StructType`

LLHD reference types are **not** in this list, causing `computeLLVMBitWidth()` to return `std::nullopt`, which triggers the verification failure.

### Code Path Analysis

1. **LowerState.cpp:219** - `ModuleLowering::run()` allocates input storage:
   ```cpp
   for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
     auto state = RootInputOp::create(allocBuilder, arg.getLoc(),
                                      StateType::get(arg.getType()), name, storageArg);
   }
   ```

2. **StateType::get()** calls `StateType::verify()` with the inner type

3. **ArcTypes.cpp** - `StateType::verify()` calls `computeLLVMBitWidth()`:
   ```cpp
   LogicalResult StateType::verify(..., Type innerType) {
     if (!computeLLVMBitWidth(innerType))
       return emitError() << "state type must have a known bit width; got " << innerType;
     return success();
   }
   ```

4. **computeLLVMBitWidth()** has no case for `llhd::RefType`:
   ```cpp
   static std::optional<uint64_t> computeLLVMBitWidth(Type type) {
     if (isa<seq::ClockType>(type)) return 1;
     if (auto intType = dyn_cast<IntegerType>(type)) return intType.getWidth();
     if (auto arrayType = dyn_cast<hw::ArrayType>(type)) { ... }
     if (auto structType = dyn_cast<hw::StructType>(type)) { ... }
     return {};  // LLHD ref types fall through to here
   }
   ```

### Why This Happens
When `circt-verilog` encounters an `inout` port, it generates LLHD's reference type (`!llhd.ref<T>`) to model the bidirectional nature. However, `arcilator` (which is designed for cycle-based simulation) cannot directly handle reference/pointer types - it expects all types to have a fixed, known bit width for state allocation.

## Hypotheses

### Hypothesis 1: Missing Type Conversion Before Arcilator (HIGH CONFIDENCE - 90%)
**Description:** The LLHD reference types from `inout` ports should be lowered/converted before reaching `arcilator`, but there's a missing conversion pass or the existing pass doesn't handle this case.

**Evidence:**
- Error occurs specifically with `!llhd.ref<i1>` type
- The `computeLLVMBitWidth()` function explicitly handles only HW/Seq dialect types
- Arcilator is designed for cycle-based simulation which doesn't natively support bidirectional signals

### Hypothesis 2: Arcilator Should Reject Unsupported Inputs Gracefully (MEDIUM CONFIDENCE - 70%)
**Description:** Arcilator should emit a user-friendly error when it encounters LLHD reference types instead of crashing with an assertion failure.

**Evidence:**
- The assertion failure provides poor error diagnostics (no source location)
- Bidirectional ports may simply be unsupported in arcilator's execution model

### Hypothesis 3: computeLLVMBitWidth Needs LLHD Type Support (MEDIUM CONFIDENCE - 60%)
**Description:** The `computeLLVMBitWidth()` function could be extended to handle LLHD reference types by extracting the inner type's bit width.

**Evidence:**
- LLHD ref types do contain an inner type with a known bit width
- However, this may not be the correct semantic interpretation for cycle-based simulation

## Keywords for Issue Search
- `StateType`
- `llhd.ref`
- `arcilator`
- `inout`
- `LowerState`
- `bit width`
- `tristate`
- `bidirectional`

## Suggested Source Files for Further Investigation

| File | Reason |
|------|--------|
| `lib/Dialect/Arc/ArcTypes.cpp` | Contains `computeLLVMBitWidth()` that fails |
| `lib/Dialect/Arc/Transforms/LowerState.cpp` | Contains failing pass |
| `include/circt/Dialect/LLHD/IR/LLHDTypes.td` | Defines LLHD ref type |
| `lib/Conversion/ExportVerilog/` | May have special handling for inout ports |
| `tools/arcilator/arcilator.cpp` | Tool entry point, may need to add validation |

## Recommendations

1. **Short-term fix:** Add a validation check in `arcilator` or `LowerStatePass` to reject modules with LLHD reference types with a clear error message indicating that bidirectional ports are not supported.

2. **Medium-term fix:** Add a lowering pass that converts LLHD reference types to a form compatible with arcilator before the `LowerStatePass` runs, or exclude such signals from simulation state.

3. **Long-term fix:** Consider if/how bidirectional signals should be modeled in arcilator's cycle-based simulation semantics.
