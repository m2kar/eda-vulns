# [MooreToCore] Assertion failure when module has string-typed output port

## Description

`circt-verilog` crashes with an assertion failure when converting a SystemVerilog module with an `output string` port type during Moore-to-Core conversion. The tool should emit a diagnostic error for unsupported types instead of asserting.

## Minimal Reproducer

```systemverilog
module example(output string str);
endmodule
```

## Command to Reproduce

```bash
circt-verilog --ir-hw bug.sv
```

## Expected Behavior

Either:
1. Emit a diagnostic error stating that string-typed ports are not supported in Moore-to-Core conversion, OR
2. Provide a valid HW type mapping for `string` ports

## Actual Behavior

The tool crashes with an assertion failure:

```
circt-verilog: /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/Casting.h:650:
decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]:
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

## Stack Trace

```
Stack dump:
 0.	Program arguments: circt-verilog --ir-hw bug.sv
 #0	llvm::sys::PrintStackTrace
 #1	llvm::sys::RunSignalHandlers
 #2	SignalHandler
 #3	/lib/x86_64-linux-gnu/libc.so.6+0x45330
 #4	(anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, ...)
 #5	(anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, ...)
 #6	mlir::OpConversionPattern<circt::moore::SVModuleOp>::matchAndRewrite
 #7	mlir::ConversionPattern::matchAndRewrite
 #8	llvm::function_ref<void ()>::callback_fn<...>
 #9	mlir::PatternApplicator::matchAndRewrite
 #10	(anonymous namespace)::OperationLegalizer::legalize
 #11	mlir::OperationConverter::convert
 #12	mlir::OperationConverter::convertOperations
 #13	void llvm::function_ref<void ()>::callback_fn<...>
 #14	applyConversion
 #15	mlir::applyFullConversion
 #16	(anonymous namespace)::MooreToCorePass::runOnOperation()
 #17	mlir::detail::OpToOpPassAdaptor::run
 #18	mlir::PassManager::run
 #19	executeWithSources
 #20	execute
 #21	main
```

## Root Cause Analysis

The Moore-to-Core conversion builds `ModulePortInfo` for module ports in `getModulePortInfo()` at `lib/Conversion/MooreToCore/MooreToCore.cpp:259`. For the `string`-typed output port, the type conversion yields an *empty/invalid* `mlir::Type` (string type is not currently supported/handled in this conversion path).

Subsequently, `ModulePortInfo::sanitizeInOut()` at `include/circt/Dialect/HW/PortImplementation.h:177` unconditionally performs `llvm::dyn_cast<circt::hw::InOutType>(type)` on this empty type, which triggers the assertion `dyn_cast on a non-existent value`.

**In summary:** An unsupported SV `string` port type is converted into a null `mlir::Type`, and later code assumes the type is valid, leading to an assertion instead of a diagnostic.

## Related Issues

This crash is a consequence of broader missing StringType support in Moore-to-Core conversion:

- **#8332** - [MooreToCore] Support for StringType from moore to llvm dialect (most relevant, tracks the core feature gap)
- **#8283** - [ImportVerilog] Cannot compile forward decleared string type (similar root cause with string type legalization)

This issue provides a concrete module-port-level crash case demonstrating the impact of the missing StringType support.

## Additional Context

- **CIRCT Version**: circt-1.139.0
- **Test Case Reduction**: 80.52% reduction from original case (231 bytes → 45 bytes)
- **Cross-Tool Validation**:
  - `verilator -sv --lint-only`: Success (valid syntax)
  - `slang --lint-only`: Build succeeded (0 errors, 0 warnings)

The test case is syntactically valid SystemVerilog but hits a missing type conversion path in CIRCT.
