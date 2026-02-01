# CIRCT Crash Reproduction Report
**Test Case ID:** 260129-000017c2

## Reproduction Status
❌ **NOT REPRODUCIBLE** - The crash has been fixed in the current toolchain

## Summary
The original crash in the arcilator tool during StateType construction has been resolved in the current CIRCT toolchain (firtool-1.139.0 with LLVM 22.0.0git).

## Original Crash Details
- **Crash Type:** Assertion failure
- **Error Message:** `state type must have a known bit width; got '!llhd.ref<i1>'`
- **Component:** arcilator - StateType::get() verification
- **Toolchain:** circt-1.139.0
- **Root Cause:** Attempting to create a StateType with !llhd.ref<i1> (a reference type instead of a concrete type)

## Test Case
**File:** source.sv
- Module: MixedPorts with inout port
- Contains: Bidirectional signal assignments and synchronous logic
- Key Feature: Mixing inout ports with registered outputs

## Reproduction Attempt
**Command Executed:**
```bash
circt-verilog --ir-hw source.sv | arcilator | opt -O0 | llc -O0 --filetype=obj
```

**Current Toolchain:**
- CIRCT firtool-1.139.0
- LLVM 22.0.0git
- Optimized build

**Result:**
✓ Successful compilation
✓ No errors or warnings
✓ Generated valid LLVM IR code

## Conclusion
The test case that previously crashed the arcilator tool now compiles successfully without any assertion failures or errors. This indicates that the bug has been fixed in the current version of the CIRCT toolchain.

## Output Files
- `reproduce.log` - Full compilation output (LLVM IR)
- `metadata.json` - Structured reproduction metadata
