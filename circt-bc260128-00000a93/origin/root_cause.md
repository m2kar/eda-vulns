# Root Cause Analysis

## Summary
`circt-verilog` asserts in `llvm::dyn_cast<circt::hw::InOutType>` while converting
`circt::moore::SVModuleOp` to HW in `MooreToCorePass`. The crash is triggered by a
module port derived from a SystemVerilog `string` input and the use of `.len()`.
During port legalization, `ModulePortInfo::sanitizeInOut()` attempts a `dyn_cast`
on a non-existent/invalid `mlir::Type` value, hitting the assertion
`"dyn_cast on a non-existent value"`.

## Evidence
- Stack shows `ModulePortInfo::sanitizeInOut()` in
  `include/circt/Dialect/HW/PortImplementation.h:177`.
- Call path originates from `getModulePortInfo()` and
  `SVModuleOpConversion::matchAndRewrite()` in
  `lib/Conversion/MooreToCore/MooreToCore.cpp`.
- Assertion: `llvm::dyn_cast(From&)` with a non-existent value.

## Suspected Trigger
The input port is declared as `string`:

```
module test(input string a, output int b);
  logic [31:0] shared_signal;
  assign shared_signal = a.len();
  assign b = shared_signal;
endmodule
```

`string` is a runtime type in SV, and the Moore dialect lowering likely produces
a port type that is either missing or not representable as a HW `InOutType`.
`sanitizeInOut()` assumes the type is present and performs `dyn_cast` without
checking for null/invalid values.

## Hypothesis
`SVModuleOpConversion` builds a `PortInfo` list where at least one entry has an
empty/invalid type for the `string` input. `ModulePortInfo::sanitizeInOut()`
does not guard against missing types and calls `dyn_cast` directly, causing the
assertion.

## Potential Fix Direction
- Guard `ModulePortInfo::sanitizeInOut()` against absent types before
  `dyn_cast`.
- In Moore-to-HW conversion, reject or explicitly lower unsupported SV `string`
  ports with a diagnostic instead of emitting invalid port types.
