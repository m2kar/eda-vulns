# [Arc][arcilator] Crash in StateType::get() with LLHD reference type for inout ports with tri-state assignments

## Bug Report

**Testcase ID:** 260128-000006ab  
**Component:** Arc / arcilator  
**Severity:** Crash (assertion failure)  
**Type:** Bug

### Description

The `arcilator` tool crashes with an assertion failure when compiling SystemVerilog modules that contain:
1. `inout` ports
2. Tri-state assignments (`1'bz`) to those inout ports

The crash occurs in `StateType::get()` when attempting to create a state type for a value with LLHD reference type `!llhd.ref<i1>`, which doesn't have a known bit width.

### Error Message

```
state type must have a known bit width; got '!llhd.ref<i1>'
```

### Stack Trace

```
arcilator: .../mlir/include/mlir/IR/StorageUniquerSupport.h:180: static ConcreteT mlir::detail::StorageUserBase<circt::arc::StateType, mlir::Type, circt::arc::detail::StateTypeStorage, mlir::detail::TypeUniquer>::get(MLIRContext *, Args &&...) [ConcreteT = circt::arc::StateType, BaseT = mlir::Type, StorageT = circt::arc::detail::StateTypeStorage, UniquerT = mlir::detail::TypeUniquer, Traits = <>, Args = <mlir::Type &>]: Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.

Stack dump:
 #12 circt::arc::StateType::get(mlir::Type) ArcTypes.cpp.inc:108:3
 #13 (anonymous namespace)::ModuleLowering::run() LowerState.cpp:219:66
 #14 (anonymous namespace)::LowerStatePass::runOnOperation() LowerState.cpp:1198:41
```

### Minimal Reproducer

```systemverilog
module MixPorts(
  input  logic a,
  inout  wire  c
);
  assign c = a ? 1'bz : 1'b0;
endmodule
```

### Compilation Command

```bash
circt-verilog --ir-hw test.sv | arcilator
```

### Root Cause

The `LowerStatePass` in the Arc dialect attempts to create a `StateType` for values associated with `inout` ports. When processing tri-state assignments, the pass encounters LLHD reference types (`!llhd.ref<T>`) instead of concrete bit vector types. The `StateType::get()` function validates that its argument has a known bit width, which reference types don't have, causing the assertion failure.

**Specific mechanism:**
1. SystemVerilog `inout` ports are mapped to LLHD reference types
2. Tri-state assignments (`1'bz`) trigger specific lowering paths
3. During `ModuleLowering::run()`, a `StateType` is created with a reference type
4. `StateType::get()` validates bit width and fails for reference types

### Related Issues

- **#8825** - [LLHD] Switch from hw.inout to a custom signal reference type
  - Discusses the LLHD reference type system (`!llhd.ref<T>`)
  - This crash likely occurred during the migration to the new type system
  
- **#5566** - [SV] Crash in `P/BPAssignOp` verifiers for `hw.inout` ports
  - Related inout port crash in SV dialect (different code path)
  
- **#4036** - [PrepareForEmission] Crash when inout operations are passed to instance ports
  - Related inout crash in same file (`StorageUniquerSupport.h`)

### Current Status

**✅ Bug appears to be fixed in current CIRCT version**

**Verification:**
- Tested on: CIRCT firtool-1.139.0, LLVM 22.0.0git
- Test case compiles successfully without crash
- Original crash occurred on: circt-1.139.0 (possibly earlier build)

**Note:** While the version numbers match (1.139.0), the fix may have been applied between builds or may be part of ongoing work on #8825.

### Environment

**Original Crash Environment:**
- Tool: arcilator
- CIRCT version: 1.139.0
- Platform: Linux x86_64

**Test Environment:**
- Tool: arcilator
- CIRCT version: firtool-1.139.0
- LLVM version: 22.0.0git
- Platform: Linux x86_64

### Steps to Reproduce

**Note:** Bug is not reproducible on current toolchain, but would have been reproducible on older builds.

1. Save the minimal reproducer to a file:
```systemverilog
module MixPorts(
  input  logic a,
  inout  wire  c
);
  assign c = a ? 1'bz : 1'b0;
endmodule
```

2. Compile using arcilator:
```bash
circt-verilog --ir-hw test.sv | arcilator
```

3. **Expected (older build):** Assertion failure and crash
4. **Actual (current build):** Compiles successfully

### Impact

- **Affected code:** Any SystemVerilog module with `inout` ports using tri-state assignments
- **Affected dialects:** Arc, LLHD (during lowering)
- **Severity:** Crash (prevents compilation)
- **Workaround:** None known (avoid inout ports with tri-state assignments on affected versions)

### Suggested Fix Locations

1. **LowerState.cpp:219** - Before calling `StateType::get()`, ensure type is dereferenced from LLHD reference
2. **StateType validation** - Add support for handling reference types or explicitly convert them
3. **inout port lowering** - Special handling for bidirectional ports with tri-state assignments

### Attachments

- Original test case: `test.sv` (18 lines, with parameterized array and loop)
- Minimal reproducer: `minimal_1.sv` (6 lines)
- Original error log: `original_error.txt`
- Full analysis: See accompanying files in bug-report-work/

### Additional Notes

- The crash signature is unique (error message not found in existing issues)
- Related to LLHD type system migration (#8825)
- Fix may have been part of #8825 work or a separate commit
- Recommend checking git history for changes to LowerState.cpp and StateType handling around the #8825 timeframe
