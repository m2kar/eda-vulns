# Root Cause Analysis - CIRCT Moore/SystemVerilog Crash (260128-000007e8)

## Crash Summary
- **Crash type**: Assertion failure
- **Assertion**: `llvm::dyn_cast` on non-existent value
- **Failure site**: `circt::hw::ModulePortInfo::sanitizeInOut()`
- **Reported location**: `include/circt/Dialect/HW/PortImplementation.h:177`
- **Call path (from stack trace)**:
  - `getModulePortInfo(...)` in `lib/Conversion/MooreToCore/MooreToCore.cpp:259`
  - `SVModuleOpConversion::matchAndRewrite(...)` in `MooreToCore.cpp:276`
  - `MooreToCorePass::runOnOperation()`

## Test Case Analysis
```systemverilog
module example(input logic clock, input logic d, output string str);
  logic q;

  always_ff @(posedge clock) begin
    q <= d;
  end

  always_comb begin
    if (q)
      str = "Hello";
    else
      str = "";
  end
endmodule
```

### Problematic Construct
- **`output string str`**: a string-typed output port on a module.

## Root Cause Hypothesis
The Moore-to-Core conversion builds `ModulePortInfo` for module ports in
`getModulePortInfo`. For the `string`-typed output port, the type conversion
appears to yield an *empty* `mlir::Type` (unsupported or unhandled SV string
type in this conversion path). `ModulePortInfo::sanitizeInOut()` then
unconditionally performs `llvm::dyn_cast<circt::hw::InOutType>(type)` on this
empty type, which triggers the assertion
`dyn_cast on a non-existent value`.

In short: **an unsupported SV `string` port type is converted into a null
`mlir::Type`, and later code assumes the type is valid**, leading to an
assertion instead of a diagnostic.

### Expected Behavior
Either:
1) Reject/diagnose unsupported SV `string` ports during Moore-to-Core
   conversion, or
2) Provide a valid HW type mapping for `string` ports, or
3) Guard `sanitizeInOut()` against null/invalid port types and emit diagnostics.

### Actual Behavior
The conversion continues with an invalid/null port type and asserts in
`sanitizeInOut()`.

## Source Availability Note
`../circt-src` was not present in the workspace. File/line references are based
on the crash log paths emitted by the CIRCT build used to reproduce the crash.
