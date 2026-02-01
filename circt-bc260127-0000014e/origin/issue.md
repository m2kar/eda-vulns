# [MooreToCore] Assertion failure with string input port: dyn_cast on a non-existent value

## Description

`circt-verilog --ir-hw` crashes with an assertion failure when processing a SystemVerilog module with a string input port during Moore-to-Core dialect conversion.

### Crash Type
**Assertion failure** in `llvm::dyn_cast<circt::hw::InOutType>` called on a null `mlir::Type`.

## Error Message

```
circt-verilog: /path/to/llvm/Support/Casting.h:650: decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]: Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

### Stack Trace (relevant frames)

```
#17 circt::hw::ModulePortInfo::sanitizeInOut()
   at include/circt/Dialect/HW/PortImplementation.h:177:24
#21 getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp)
   at lib/Conversion/MooreToCore/MooreToCore.cpp:259:1
#22 SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, ...)
   at lib/Conversion/MooreToCore/MooreToCore.cpp:276:32
```

## Minimal Test Case

```systemverilog
module test(input string a, output int b);
endmodule
```

### Expected Behavior
Either:
1. Properly convert string input ports to a supported HW representation, OR
2. Emit a clear diagnostic that string ports are currently unsupported

### Actual Behavior
Compiler crashes with assertion failure during Moore-to-Core lowering.

## Reproduction Steps

```bash
# Using the minimal test case
export PATH=/opt/llvm-22/bin:$PATH
circt-verilog --ir-hw bug.sv
```

**Command that triggers crash:**
```bash
circt-verilog --ir-hw bug.sv
```

## Root Cause Analysis

The Moore-to-Core `TypeConverter` does not have a conversion rule for the Moore `string` type. When processing an SV module with a string input port:

1. `getModulePortInfo` calls `convertType()` on the string port type
2. The type converter returns a **null `mlir::Type`** (unsupported type)
3. `PortInfo` is constructed with this null type
4. `ModulePortInfo::sanitizeInOut()` is called on the port list
5. `sanitizeInOut()` attempts `dyn_cast<hw::InOutType>` on each port type
6. The null type causes the assertion `detail::isPresent(Val)` to fail

### Affected Code Components

- `lib/Conversion/MooreToCore/MooreToCore.cpp:getModulePortInfo` (line 259)
- `lib/Conversion/MooreToCore/MooreToCore.cpp:SVModuleOpConversion::matchAndRewrite`
- `include/circt/Dialect/HW/PortImplementation.h:ModulePortInfo::sanitizeInOut` (line 177)
- Moore-to-Core TypeConverter (missing string type conversion)

### Suggested Fix

1. **Add type validation**: In `getModulePortInfo`, check if `convertType()` returns null and emit a diagnostic before constructing `PortInfo`
2. **Add string type conversion**: Either lower Moore `string` to a supported HW representation or explicitly reject it with a clear error
3. **Defensive guard**: Add null checks in `sanitizeInOut()` before calling `dyn_cast` (though fixing the root cause is preferred)

## Environment

- **CIRCT Version**: firtool-1.139.0
- **LLVM Version**: 22.0.0git
- **Tool**: `circt-verilog --ir-hw`
- **Testcase ID**: 260127-0000014e

## Additional Context

### Validation Results
- **Syntax**: Valid (verified by `circt-verilog --parse-only`, slang, and verilator)
- **Cross-tools**: Passes with slang and verilator lint-only
- **Classification**: Bug report (valid SystemVerilog, compiler crash)

### Related Issues
- Issue #8219: Similar assertion pattern in ESI dialect (different root cause - bundle handling vs string type conversion)
- Issue #7628: MooreToCore string constants support (strings in general, but not input port handling)

### Reduction Summary
- **Original test case**: 7 lines with wire declarations and assignments
- **Minimized test case**: 2 lines (71.43% reduction)
- Both trigger the same crash, confirming the string input port is the root trigger

## Checklist

- [x] Reproduced with current CIRCT toolchain
- [x] Minimized test case provided
- [x] Root cause analysis completed
- [x] Checked for duplicate issues (top match: Issue #8219, score 5/15, different root cause)
- [x] Validation confirms valid SystemVerilog syntax
