# [MooreToCore] Crash when using `string` type as module port

## Description

`circt-verilog` crashes with an assertion failure when processing a SystemVerilog module that uses `string` type as a module port.

## Minimal Test Case

```systemverilog
module m(output string s);
endmodule
```

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv
```

## Expected Behavior

Either:
1. Successfully convert the `string` type to a target representation (e.g., LLVM pointer), OR
2. Emit a proper diagnostic error explaining that `string` ports are not supported

## Actual Behavior

```
circt-verilog: /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/Casting.h:650:
decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]:
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.

PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
Stack dump:
0.	Program arguments: circt-verilog --ir-hw bug.sv
 #12 llvm::SmallVector<circt::hw::PortInfo, 1u>::~SmallVector() llvm/.../SmallVector.h:1207:18
 #13 (anonymous namespace)::getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp) lib/Conversion/MooreToCore/MooreToCore.cpp:259:1
 #14 (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, ...) lib/Conversion/MooreToCore/MooreToCore.cpp:276:32
 #35 (anonymous namespace)::MooreToCorePass::runOnOperation() lib/Conversion/MooreToCore/MooreToCore.cpp:2571:7
 #36 executeWithSources(mlir::MLIRContext*, llvm::SourceMgr&) tools/circt-verilog/circt-verilog.cpp:398:9
Aborted (core dumped)
```

## Root Cause Analysis

The `MooreToCore` conversion pass lacks a type conversion rule for `moore::StringType`. The function `populateTypeConversion()` in `MooreToCore.cpp` (lines 2268-2410) defines conversions for various Moore types (IntType, RealType, StructType, etc.) but does **not** include `StringType`.

When processing a module port with `string` type:
1. `typeConverter.convertType(StringType)` returns `Type()` (null type)
2. The `getModulePortInfo()` function attempts to handle this
3. During error handling cleanup, `SmallVector<hw::PortInfo>` destructor runs
4. The `dyn_cast<hw::InOutType>` on a null type triggers the assertion failure

## Related Issues

### #8930 - [MooreToCore] Crash with sqrt/floor
- **Similarity**: Same assertion failure (`dyn_cast on a non-existent value`)
- **Difference**: Triggered by `real` type instead of `string` type
- **Connection**: Both represent missing type conversion rules in MooreToCore

### #8332 - [MooreToCore] Support for StringType from moore to llvm dialect
- **Similarity**: Direct feature request for `StringType` support
- **Connection**: This crash is the direct consequence of the missing feature discussed in #8332
- **Recommendation**: Consider adding this test case to #8332 as additional evidence

### Why This Issue?
While #8930 and #8332 are highly related, this issue specifically:
1. Documents the **minimal reproducible case** for `string` as a module port
2. Shows that the crash occurs even with the simplest possible module (1 port, 0 internal logic)
3. Provides a distinct trigger (`string` type) separate from #8930 (`real` type)
4. Can serve as a regression test when `StringType` support is implemented

## Validation

- **Syntax Check**: ✅ Valid (verified with slang v10.0.6)
- **Cross-Tool**: ✅ Verilator lint passes
- **IEEE 1800**: `string` type is a valid SystemVerilog data type (Section 6.16)

The test case is valid IEEE 1800 SystemVerilog. CIRCT should either support `string` type conversion or emit a clear diagnostic error instead of crashing.

## Version Information

```
circt-verilog --version
LLVM 22.0.0git (firtool-1.139.0)
```

## Suggested Fix

1. **Add type conversion rule for `StringType`** in `populateTypeConversion()`:
   ```cpp
   typeConverter.addConversion([&](StringType type) -> std::optional<Type> {
     // Map to LLVM pointer (similar to ChandleType)
     return LLVM::LLVMPointerType::get(type.getContext());
   });
   ```

2. **Or**: Reject with proper error message:
   ```cpp
   typeConverter.addConversion([&](StringType type) -> std::optional<Type> {
     // Emit clear error
     return std::nullopt;
   });
   ```

## Additional Context

- The crash occurs during error handling cleanup in `getModulePortInfo()`
- This suggests a broader issue: error paths should not trigger additional assertions
- `string` as internal variables might work in some contexts, but module ports require explicit type conversion

## References

- Related: #8930 (same assertion, different type)
- Related: #8332 (StringType feature request)
- Crash location: `lib/Conversion/MooreToCore/MooreToCore.cpp:259`
- Failing pass: `MooreToCorePass`
