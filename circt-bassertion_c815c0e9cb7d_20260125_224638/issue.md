<!-- Title: [Moore] Assertion failure in MooreToCore when module has string-typed ports -->

## Description

CIRCT `circt-verilog` crashes with an assertion failure in `dyn_cast` when processing a SystemVerilog module with `string` type ports. The crash occurs in the MooreToCore conversion pass when attempting to create `hw::ModulePortInfo`. The underlying issue is that `sim::DynamicStringType` (the converted type from `moore::StringType`) triggers an assertion when `sanitizeInOut()` calls `dyn_cast<hw::InOutType>` on a non-existent/invalid type value.

**Crash Type**: assertion
**Dialect**: Moore
**Failing Pass**: MooreToCore

## Steps to Reproduce

1. Save test case below as `test.sv`
2. Run:
   ```bash
   circt-verilog --ir-hw test.sv
   ```

## Test Case

```systemverilog
module top_module(output string str_out);
endmodule
```

## Error Output

```
PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
Stack dump:
0.	Program arguments: circt-verilog --ir-hw bug.sv
 #4 0x00007f705356b8ae (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, circt::moore::SVModuleOpAdaptor, mlir::ConversionPatternRewriter&) const MooreToCore.cpp:0:0
 #16 0x00007f705353d231 (anonymous namespace)::MooreToCorePass::runOnOperation() MooreToCore.cpp:0:0
...
```

## Root Cause Analysis

### Hypothesis 1 (High Confidence)
**Cause**: The type converter returns `sim::DynamicStringType` for `moore::StringType`, but this type may not be valid in the context where `hw::ModulePortInfo::sanitizeInOut()` is called, leading to an assertion failure in `dyn_cast`.

**Evidence**:
- The assertion message indicates `dyn_cast` was called on a "non-existent value"
- `sim::DynamicStringType` is not part of the `hw` dialect type system
- The stack trace shows the crash occurs in `getModulePortInfo` → `ModulePortInfo` constructor

**Mechanism**:
1. `moore::StringType` is converted to `sim::DynamicStringType`
2. This converted type is placed into `hw::PortInfo`
3. When `sanitizeInOut()` checks `dyn_cast<hw::InOutType>(p.type)`, the MLIR type system may reject this as an invalid type for this context
4. The `detail::isPresent(Val)` check fails, triggering the assertion

### Suggested Fix Directions

1. **Add null-check after type conversion** (Recommended):
   ```cpp
   Type portTy = typeConverter.convertType(port.type);
   if (!portTy) {
     // Emit error or handle unsupported type
     return failure();
   }
   ```

2. **Use `dyn_cast_if_present` instead of `dyn_cast` in sanitizeInOut()**:
   ```cpp
   if (auto inout = dyn_cast_if_present<hw::InOutType>(p.type)) {
   ```

3. **Validate port types before module creation** - Reject unsupported types like `string` at an earlier stage with a proper error message.

4. **Extend HW dialect support** - If `string` ports should be supported, ensure proper lowering path exists.

## Environment

- **CIRCT Version**: firtool-1.139.0
- **OS**: Linux
- **Architecture**: x86_64

## Stack Trace

<details>
<summary>Click to expand stack trace</summary>

```
 #4 0x00007f705356b8ae (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, circt::moore::SVModuleOpAdaptor, mlir::ConversionPatternRewriter&) const MooreToCore.cpp:0:0
 #5 0x00007f705356bb93 llvm::LogicalResult mlir::ConversionPattern::dispatchTo1To1<mlir::OpConversionPattern<circt::moore::SVModuleOp>, circt::moore::SVModuleOp>(mlir::OpConversionPattern<circt::moore::SVModuleOp> const&, circt::moore::SVModuleOp, circt::moore::SVModuleOp::GenericAdaptor<llvm::ArrayRef<mlir::ValueRange>>, mlir::ConversionPatternRewriter&) (/opt/firtool-1.139.0/bin/../lib/libCIRCTMooreToCore.so+0x50b93)
 #16 0x00007f705353d231 (anonymous namespace)::MooreToCorePass::runOnOperation() MooreToCore.cpp:0:0
```

</details>

## Related Issues

- #8283: [ImportVerilog] Cannot compile forward declared string type
- #8332: [MooreToCore] Support for StringType from moore to llvm dialect
- #8930: [MooreToCore] Crash with sqrt/floor
- #8173: [ImportVerilog] Crash on ordering-methods-reverse test
- #4036: [PrepareForEmission] Crash when inout operations are passed to instance ports

---

*This issue was generated with assistance from an automated bug reporter.*
