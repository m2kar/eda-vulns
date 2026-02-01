# Root Cause Analysis

## Summary
The crash is an assertion in `llvm::dyn_cast` when `circt::hw::ModulePortInfo::sanitizeInOut()` attempts to cast a
non-existent `mlir::Type`. This happens while lowering a Moore SystemVerilog module to HW/Core, and is triggered by a
module port declared as an unpacked array of `string` (`output string s[1:0]`). The Moore-to-Core type conversion likely
returns a null/empty type for this unsupported port type, which then flows into `sanitizeInOut()` without a guard.

## Stack Trace Analysis
- **#17** `circt::hw::ModulePortInfo::sanitizeInOut()` at
  `include/circt/Dialect/HW/PortImplementation.h:177` triggers the assertion (`dyn_cast` on a non-existent value).
- **#21** `(anonymous namespace)::getModulePortInfo()` at
  `lib/Conversion/MooreToCore/MooreToCore.cpp:259` builds the `PortInfo` list that is later sanitized.
- **#22** `(anonymous namespace)::SVModuleOpConversion::matchAndRewrite()` at
  `lib/Conversion/MooreToCore/MooreToCore.cpp:276` calls `getModulePortInfo()` during the Moore→Core conversion.

## Test Case Pattern Analysis
Key constructs in `source.sv`:
- `output string s[1:0]`: an **unpacked array of SystemVerilog strings** as a module port.
- Assignments to `s` from `initial` and `always_ff` blocks.

Moore-to-Core/HW lowering generally expects hardware-compatible types. `string` (and especially an unpacked array of
`string`) is typically not representable as an HW port type, so the type conversion likely fails and returns an empty
`mlir::Type` for this port.

## Root Cause Hypothesis
During Moore-to-Core lowering, `getModulePortInfo()` calls the type converter on each port. For `output string s[1:0]`,
the converter returns a null/empty `mlir::Type` because `string`/unpacked arrays are unsupported as HW ports. The
conversion code still records the port and then calls `ModulePortInfo::sanitizeInOut()`, which blindly performs
`dyn_cast<circt::hw::InOutType>` on the empty type. This triggers the assertion:
`dyn_cast on a non-existent value`.

## Relevant Code Locations (from stack trace)
- `include/circt/Dialect/HW/PortImplementation.h:177` — `circt::hw::ModulePortInfo::sanitizeInOut()`
- `lib/Conversion/MooreToCore/MooreToCore.cpp:259` — `getModulePortInfo()`
- `lib/Conversion/MooreToCore/MooreToCore.cpp:276` — `SVModuleOpConversion::matchAndRewrite()`

## Potential Fix Suggestions
1. **Fail fast on unsupported port types**: In `getModulePortInfo()`, check the result of `convertType`. If it returns
   null, emit a diagnostic on the port and return failure rather than constructing a `PortInfo` entry.
2. **Add defensive checks in `sanitizeInOut()`**: Guard against null/empty types before `dyn_cast`, and emit a clearer
   diagnostic that includes the port name/type.
3. **Extend type conversion** (if intended): Implement lowering for `string`/unpacked string arrays to a supported HW
   representation, or explicitly mark them as illegal with a diagnostic.

*Note: CIRCT source is not present locally; line numbers are derived from the crash stack trace.*
