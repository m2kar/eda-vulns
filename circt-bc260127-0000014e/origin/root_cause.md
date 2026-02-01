# Root Cause Analysis

## Crash summary
- **Crash type:** Assertion failure in `llvm::dyn_cast` on a non-existent value.
- **Invocation:** `circt-verilog --ir-hw` during Moore-to-Core lowering.
- **Crash location:** `include/circt/Dialect/HW/PortImplementation.h:177` in
  `circt::hw::ModulePortInfo::sanitizeInOut()`, called from
  `lib/Conversion/MooreToCore/MooreToCore.cpp:259` (`getModulePortInfo`).
- **Failed assertion:** `detail::isPresent(Val) && "dyn_cast on a non-existent value"`.
- **Note:** `../circt-src` is not present in this workspace; analysis is based on the
  stack trace and known CIRCT lowering behavior.

## Code analysis
The stack trace shows Moore-to-Core conversion building module port information and
then calling `ModulePortInfo::sanitizeInOut()`. That helper walks the port list and
uses `dyn_cast<circt::hw::InOutType>` on each port type when normalizing inout
ports. The assertion indicates the `mlir::Type` passed to `dyn_cast` is null
(`mlir::Type()`), not merely of the wrong kind.

In `getModulePortInfo`, port types are derived via a `TypeConverter`. If the
converter cannot handle a type, it returns a null `mlir::Type`. The current path
does not appear to guard against a failed conversion before calling
`sanitizeInOut()`, leading to a null type being fed into `dyn_cast` and the
assertion firing.

## Test case analysis
```systemverilog
module test(input string a, output int b);
  wire [31:0] len_signal = a.len();
  wire [31:0] inverted_len;
  assign inverted_len = ~len_signal;
  assign b = inverted_len;
endmodule
```

**Features used:**
- **String input port** (`input string a`).
- **String method call** (`a.len()` returns the string length as an integer).
- **Packed bit vector wires** (`wire [31:0]`), **bitwise NOT**, and **continuous
  assignments**.

**Expected behavior:** the compiler should either lower string ports and the
`.len()` method into supported core/HW types or emit a clean diagnostic that string
ports are unsupported. It should not assert.

## Root cause hypothesis
The input port’s **string type** is not supported by the Moore-to-Core
`TypeConverter`, so `convertType` returns a null `mlir::Type`. `getModulePortInfo`
still constructs `PortInfo` with this null type and calls
`ModulePortInfo::sanitizeInOut()`, which blindly `dyn_cast`s the type to
`hw::InOutType`. Because the type is null, the `dyn_cast` assertion triggers.

This is most likely a **missing type conversion or missing diagnostic/validation**
for string ports during Moore-to-Core lowering, rather than a deeper type system
inconsistency.

## Suggested fix approach
1. **Handle unsupported types earlier:** In `getModulePortInfo` (or the Moore
   `TypeConverter`), check the result of `convertType`. If null, emit a diagnostic
   and fail the conversion instead of proceeding.
2. **Add explicit conversion or rejection for `string`:** Either lower `string`
   to a supported HW representation (if that is intended) or emit a clear error
   that string ports are unsupported.
3. **Defensive guard:** Consider asserting or early-returning on null port types
   before calling `sanitizeInOut()` to avoid `dyn_cast` on null values.
