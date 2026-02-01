**DUPLICATE NOTICE**: This issue appears to be a duplicate of Issue #9572
(similarity score: 9.8/10). This report is generated for review purposes.

Original duplicate issue: https://github.com/llvm/circt/issues/9572

# [Moore] Assertion failure when module has string type output port

## Description
The `circt-verilog` tool crashes with an assertion failure when processing a SystemVerilog module that contains an `output string` port. The crash occurs during the `MooreToCore` conversion pass. Specifically, the `TypeConverter` used in `MooreToCore.cpp` fails to provide a conversion rule for the `moore.string` type, resulting in a null/empty type being passed to `ModulePortInfo`. This subsequently triggers a `dyn_cast` on a non-existent value in `ModulePortInfo::sanitizeInOut`.

## Reproduction Steps
1. Create a SystemVerilog file `bug.sv` with a module containing an `output string` port.
2. Run `circt-verilog --ir-hw bug.sv`.

## Minimal Test Case
```systemverilog
module top(output string out); endmodule
```

## Reproduction Command
```bash
circt-verilog --ir-hw bug.sv
```

## Expected Behavior
The tool should either correctly convert the `string` type (e.g., to `sim.dynamic_string`) or emit a clear error message stating that the port type is unsupported, instead of crashing.

## Actual Behavior
The tool crashes with an assertion failure.

## Error Message
```
PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
Stack dump:
0.	Program arguments: circt-verilog --ir-hw bug.sv
 #0 0x00007f5de585d8a8 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int)
 #1 0x00007f5de585b2f5 llvm::sys::RunSignalHandlers()
 #2 0x00007f5de585e631 SignalHandler(int, siginfo_t*, void*)
 #3 0x00007f5de536b330
 #4 0x00007f5de9a7f8ae (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, circt::moore::SVModuleOpAdaptor, mlir::ConversionPatternRewriter&) const MooreToCore.cpp:0:0
 ...
```
Full error trace indicates: `Assertion 'detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.` at `llvm/include/llvm/Support/Casting.h:650`.

## Root Cause Analysis
The crash is caused by a missing type conversion rule for `moore::StringType` in the `MooreToCore` pass's `TypeConverter`.

1. In `MooreToCore.cpp`, `getModulePortInfo` iterates through the ports of the module.
2. It calls `typeConverter.convertType(port.type)` for each port.
3. For a `string` type port, the converter returns a null `mlir::Type` because no conversion rule is defined for `moore.string`.
4. This null type is used to construct a `hw::PortInfo` object, which is then added to a `hw::ModulePortInfo` collection.
5. The `hw::ModulePortInfo` constructor calls `sanitizeInOut()`.
6. Inside `sanitizeInOut()`, it attempts to `dyn_cast<hw::InOutType>(p.type)`.
7. Since `p.type` is null, the `dyn_cast` (which requires a non-null value in this context) triggers the assertion: `dyn_cast on a non-existent value`.

## Environment Information
- **CIRCT Version**: firtool-1.139.0
- **Dialect**: moore (SystemVerilog)
- **Target Dialect**: hw
- **OS**: Linux
