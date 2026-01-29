# [MooreToCore] Assertion failure when using string type as module output port

## Description

`circt-verilog` crashes with an assertion failure when processing a SystemVerilog module that declares a `string` type output port. The crash occurs in `MooreToCore` conversion pass when `ModulePortInfo::sanitizeInOut()` attempts to perform a `dyn_cast<hw::InOutType>` on `sim::DynamicStringType`, which is not a valid HW dialect type.

## Minimal Test Case

```systemverilog
module example(
  output string str
);
endmodule
```

## Reproduction Steps

1. Save the test case as `bug.sv`
2. Run: `circt-verilog --ir-hw bug.sv`

## Expected Behavior

The compiler should either:
1. Successfully process the code, or
2. Emit a clear diagnostic message if string type ports are unsupported

## Actual Behavior

```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
PLEASE submit a bug report to https://github.com/llvm/circt and include crash backtrace.
Stack dump:
 0.	Program arguments: /edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw bug.sv
 #13 getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp)
     @ MooreToCore.cpp:259
 #14 SVModuleOpConversion::matchAndRewrite(...)
     @ MooreToCore.cpp:276
 #35 MooreToCorePass::runOnOperation()
     @ MooreToCore.cpp:2571
Aborted (core dumped)
```

## Root Cause

The crash occurs in the following call sequence:

1. **Parsing**: `output string str` is parsed and a Moore `StringType` port is created
2. **Type Conversion**: MooreToCore converts `StringType` → `sim::DynamicStringType`
3. **Port Info**: `getModulePortInfo()` creates `PortInfo` with `sim::DynamicStringType`
4. **Sanitization**: `ModulePortInfo` constructor calls `sanitizeInOut()`
5. **Crash**: `dyn_cast<hw::InOutType>(sim::DynamicStringType)` fails because:
   - `sim::DynamicStringType` is NOT in `isHWValueType()` accepted types
   - The `dyn_cast` receives an invalid/null type
   - Assertion `detail::isPresent(Val)` fails

The `sanitizeInOut()` function in `PortImplementation.h` assumes all port types are valid HW dialect types, but `sim::DynamicStringType` (used for simulation) is not compatible with HW dialect port handling.

## Validation

| Tool | Version | Result |
|------|---------|--------|
| Verilator | 5.022 | ✅ Pass |
| Slang | 10.0.6 | ✅ Pass |
| CIRCT | 1.139.0 | ❌ Crash |

The test case follows IEEE 1800-2017 standard (string is a built-in type, §6.16). Other SystemVerilog tools handle it correctly.

## Version

- CIRCT: 1.139.0
- Built from: circt-1.139.0-src

## Suggested Fixes

1. **Add validation in `getModulePortInfo()`** (MooreToCore.cpp):
   ```cpp
   Type portTy = typeConverter.convertType(port.type);
   if (!portTy || !hw::isHWValueType(portTy)) {
     // Emit error: unsupported port type
     return failure();
   }
   ```

2. **Reject string ports during Moore parsing**:
   - Add diagnostic: "string type cannot be used as module port in synthesis flow"

3. **Make `sanitizeInOut()` defensive** (PortImplementation.h):
   ```cpp
   void sanitizeInOut() {
     for (auto &p : ports)
       if (p.type && isa<hw::InOutType>(p.type)) {
         // ...
       }
   }
   ```

## Related Issues

- #8930 - Same assertion in MooreToCore, but different trigger (sqrt/floor vs string port)
- #8332 - Discussion on StringType support in MooreToCore
- #8283 - String variable compilation issue

## Keywords

MooreToCore, string, DynamicStringType, ModulePortInfo, sanitizeInOut, InOutType, assertion, port type

## Crash Signature

```
Assertion: "dyn_cast on a non-existent value"
Location: MooreToCore.cpp:259 in getModulePortInfo()
Failing Pass: MooreToCorePass
Dialect: Moore
```
