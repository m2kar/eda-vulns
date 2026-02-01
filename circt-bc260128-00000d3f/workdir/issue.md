# [MooreToCore] Crash with packed union port type (dyn_cast assertion)

## Bug Description
The `circt-verilog` tool crashes with an assertion failure when a module port uses a `packed union` type during the `MooreToCore` conversion pass. The crash occurs because the `TypeConverter` in `MooreToCore` does not yet support `packed union` types, returning a null type which is subsequently used in a `dyn_cast` within `circt::hw::ModulePortInfo::sanitizeInOut()`.

## Minimal Testcase
```systemverilog
// Minimal test case: packed union as module port crashes circt-verilog
typedef union packed { logic a; } u;
module m(input u i);
endmodule
```

## Reproduction Command
```bash
/opt/firtool/bin/circt-verilog --ir-hw bug.sv
```

## Error Output
```
circt-verilog: llvm/llvm/include/llvm/Support/Casting.h:650: decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]: Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

### Stack Dump (Partial)
```
 #16 0x000055b63ada1874 decltype(auto) llvm::dyn_cast<circt::hw::InOutType, mlir::Type>(mlir::Type&)
 #17 0x000055b63ada1874 circt::hw::ModulePortInfo::sanitizeInOut() at include/circt/Dialect/HW/PortImplementation.h:177
 #21 0x000055b63b099753 (anonymous namespace)::getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp) at lib/Conversion/MooreToCore/MooreToCore.cpp:259
 #22 0x000055b63b099753 (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, circt::moore::SVModuleOpAdaptor, mlir::ConversionPatternRewriter&) const at lib/Conversion/MooreToCore/MooreToCore.cpp:276
```

## Root Cause Analysis
The crash is caused by the lack of support for `packed union` types in the `MooreToCore` type converter.

1. In `MooreToCore.cpp`, `getModulePortInfo` calls `typeConverter.convertType(port.type)`.
2. Since `packed union` conversion is not implemented, it returns `nullptr`.
3. The code proceeds to create a `hw::PortInfo` with this null type.
4. When `hw::ModulePortInfo` is constructed, it calls `sanitizeInOut()`.
5. `sanitizeInOut()` performs a `dyn_cast<hw::InOutType>(p.type)` on the null type, triggering the LLVM assertion.

## Validation
### Syntax Compliance
The testcase is valid SystemVerilog according to IEEE 1800-2017 Section 7.3.1.
- **Verilator 5.022**: Pass (No errors)
- **Slang 10.0.6**: Pass (0 errors, 0 warnings)

### Cross-tool Behavior
| Tool | Result |
|------|--------|
| Verilator | Successfully parsed |
| Slang | Successfully parsed |
| CIRCT | **Crash** (Assertion Failure) |

## Related Issues
This crash follows a systemic pattern of missing null checks after type conversion in CIRCT passes, similar to:
- #8219: [ESI] Assertion: dyn_cast on a non-existent value
- #8930: [MooreToCore] Crash with sqrt/floor
