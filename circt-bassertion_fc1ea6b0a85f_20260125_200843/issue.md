# [FIXED] arcilator crashes with assertion when processing inout ports

## Summary

`arcilator` crashes with an assertion failure when processing modules containing `inout` (bidirectional) ports. The crash occurs in the `LowerState` pass when attempting to create an `arc::StateType` for an `!llhd.ref<i1>` type, which represents inout ports in the HW dialect.

**Status**: This bug has been **fixed** in current CIRCT versions. The issue is being reported for historical documentation purposes.

## Bug Information

| Field | Value |
|--------|-------|
| **Tool** | arcilator |
| **Command** | `circt-verilog --ir-hw file.sv \| arcilator` |
| **Dialect** | Arc |
| **Failing Pass** | LowerState (`arc-lower-state`) |
| **Crash Type** | Assertion failure |
| **Original Version** | CIRCT 1.139.0 |
| **Current Version** | LLVM 22.0.0git (fixed) |
| **Issue Number** | [Not assigned - documentation only] |

## Minimal Test Case

```systemverilog
module M(inout logic c);
endmodule
```

**Original crash command**:
```bash
circt-verilog --ir-hw bug.sv | arcilator
```

## Error Details

### Original Error Message
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
```

### Assertion Failure
```
arcilator: .../mlir/include/mlir/IR/StorageUniquerSupport.h:180:
static ConcreteT mlir::detail::StorageUserBase<...>::get(MLIRContext *, Args &&...) [with ConcreteT = circt::arc::StateType, ...]:
Assertion `succeeded(ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### Stack Trace (Key Frames)
```
#11 circt::arc::StateType::get(mlir::Type)
    ArcTypes.cpp.inc:108
#12 (anonymous namespace)::ModuleLowering::run()
    LowerState.cpp:219
#14 (anonymous namespace)::LowerStatePass::runOnOperation()
    LowerState.cpp:1198
```

## Root Cause Analysis

### Problem Mechanism

1. `circt-verilog --ir-hw` converts SystemVerilog inout ports to `!llhd.ref<T>` types in the HW dialect
2. `arcilator` runs the `LowerState` pass on the converted IR
3. The pass iterates over module arguments to allocate storage:
   ```cpp
   // lib/Dialect/Arc/Transforms/LowerState.cpp:219
   for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
     auto state = StateType::get(arg.getType());  // arg.getType() = !llhd.ref<i1>
     // ^^^ Validation fails here
   }
   ```
4. `StateType::verify()` calls `computeLLVMBitWidth(innerType)`
5. `computeLLVMBitWidth()` doesn't recognize `llhd.ref` types and returns `std::nullopt`
6. Validation fails → assertion triggers → crash

### Design Limitation

Arc dialect does not support inout ports by design (confirmed in `ArcOps.cpp:338-339`):
```cpp
for (const hw::ModulePort &port : getIo().getPorts())
  if (port.dir == hw::ModulePort::Direction::InOut)
    return emitOpError("inout ports are not supported");
```

However, this check occurs too late in the pipeline - after `LowerState` has already attempted to create `StateType` for the unsupported port.

### Root Cause

**Missing early validation**: The `arcilator` pipeline lacks early detection of unsupported `inout` ports, leading to an assertion failure instead of a clear error message.

## Classification

| Field | Value |
|-------|-------|
| **Bug Type** | Missing early validation / Assertion instead of error |
| **Severity** | Medium |
| **Component** | Arc dialect / arcilator |
| **Is Genuine Bug** | ✅ Yes (historical) |
| **Unsupported Feature** | ✅ Yes (inout ports) |
| **Current Status** | ✅ Fixed |

## Validation Results

| Check | Result |
|-------|--------|
| **Test Case Validity** | ✅ Valid SystemVerilog syntax |
| **Reproduction** | ❌ Bug fixed in current version |
| **Cross-tool Verification** | ✅ slang: pass, verilator: pass |
| **Current Behavior** | ✅ arcilator handles inout ports gracefully |

**Note**: The current CIRCT version (LLVM 22.0.0git) successfully processes the test case without crashing, indicating the bug has been fixed. The tool now handles inout ports more gracefully (likely by filtering them out or adding early validation).

## Suggested Fix (for Historical Reference)

### Option 1: Early Detection (Recommended)
Add early validation in the `arcilator` pipeline before the `LowerState` pass:

```cpp
// At arcilator entry or before LowerState
for (auto port : module.getPorts()) {
  if (port.dir == hw::ModulePort::Direction::InOut) {
    return module.emitError()
           << "inout ports are not supported by arcilator";
  }
}
```

### Option 2: Graceful Error Handling
Modify `LowerState.cpp` to handle `StateType::get()` failures gracefully:

```cpp
auto stateType = StateType::getChecked(
    [&]() { return emitError(arg.getLoc()); },
    arg.getType());
if (!stateType)
  return failure();  // Instead of assertion
```

## Duplicate Check

| Query | Results | Top Match |
|-------|----------|-----------|
| "arcilator inout" | Found 6 issues | #4916 (5.5 score, different issue) |
| "StateType \"known bit width\"" | No results | N/A |
| "llhd.ref" | Found 1 issue | #8825 (5.0 score, LLHD design discussion) |

**Conclusion**: No duplicate issue found. Related issue #8825 discusses `!llhd.ref<T>` type design in the LLHD dialect but is a feature request, not a bug report about arcilator crashes.

## Files Referenced

- `lib/Dialect/Arc/Transforms/LowerState.cpp` - Crash location
- `lib/Dialect/Arc/ArcTypes.cpp` - StateType verification logic
- `lib/Dialect/Arc/ArcOps.cpp` - ModelOp inout port check
- `tools/arcilator/arcilator.cpp` - Tool entry point

## Reports

- `root_cause.md` - Detailed root cause analysis
- `analysis.json` - Structured analysis data
- `validation.md` - Validation report
- `duplicates.md` - Duplicate check report
- `minimize_report.md` - Test case minimization report

## Keywords

`arcilator`, `inout`, `StateType`, `llhd.ref`, `LowerState`, `known bit width`, `bidirectional port`, `assertion`
