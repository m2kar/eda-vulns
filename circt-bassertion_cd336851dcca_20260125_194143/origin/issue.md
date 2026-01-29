# [MooreToCore] Crash on module with `output string` port - Assertion `dyn_cast on a non-existent value`

## Issue Summary

CIRCT crashes with an assertion failure when compiling a SystemVerilog module that has a port declared with `string` type. The crash occurs in the `MooreToCore` conversion pass because the type converter lacks a handler for `moore::StringType`.

### Error Message
```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed
```

### Crash Location
```
lib/Conversion/MooreToCore/MooreToCore.cpp:259
function: (anonymous namespace)::getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp)
```

---

## Reduced Test Case

```systemverilog
module test_module(output string result);
endmodule
```

This is the minimal test case that reproduces the crash. The original test case had 13 lines; it has been reduced by **84.6%**.

## Reproduction Steps

1. Save the test case to a file (e.g., `test.sv`)
2. Run the following command:
   ```bash
   circt-verilog --ir-hw test.sv
   ```

### Expected Behavior
The compiler should either:
- Properly support `string` type ports in MooreToCore conversion, OR
- Emit a clear diagnostic message that string ports are not supported

### Actual Behavior
The compiler crashes with an assertion failure.

---

## Root Cause Analysis

### Summary
The `MooreToCore` pass's `populateTypeConversion()` function registers type converters for various Moore dialect types (`IntType`, `RealType`, `ArrayType`, `FormatStringType`, etc.), but **no converter for `StringType`**.

When `getModulePortInfo()` processes module ports and calls `typeConverter.convertType(port.type)` for a `string` typed port:
1. The converter returns `null` (no matching converter found)
2. The null type propagates to `hw::PortInfo`
3. Later, when `ModulePortInfo::sanitizeInOut()` attempts `dyn_cast<hw::InOutType>` on the null type
4. The assertion fails because the type is not present

### Crash Stack Trace (Key Frames)
```cpp
#21 getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp)
    at lib/Conversion/MooreToCore/MooreToCore.cpp:259

#22 SVModuleOpConversion::matchAndRewrite(...)
    at lib/Conversion/MooreToCore/MooreToCore.cpp:276

#17 ModulePortInfo::sanitizeInOut()
    at include/circt/Dialect/HW/PortImplementation.h:177

#16 llvm::dyn_cast<hw::InOutType, mlir::Type>(mlir::Type&)
    at llvm/Support/Casting.h:651
```

### Root Cause Hypotheses (Confidence)

#### Hypothesis 1 (High Confidence)
**Cause**: `populateTypeConversion()` in `MooreToCore.cpp` lacks a type converter for `moore::StringType`

**Evidence**:
- Test case uses `output string result` port
- `populateTypeConversion()` has converters for `IntType`, `RealType`, `ArrayType`, `FormatStringType`, but **not `StringType`**
- `FormatStringType` (for `$display` format strings) has a converter mapping to `sim::FormatStringType`
- Null type from `convertType()` propagates to `hw::PortInfo`, causing downstream `dyn_cast` failure

#### Hypothesis 2 (Medium Confidence)
**Cause**: String type as module port may be intentionally unsupported but lacks proper error handling

**Evidence**:
- String is a dynamic type typically used in testbenches
- Hardware modules should have fixed-width signal types
- Missing diagnostic leads to crash instead of clear error message

#### Hypothesis 3 (Low Confidence)
**Cause**: StringType should map to simulation-specific or LLVM pointer type like ChandleType

**Evidence**:
- `FormatStringType` maps to `sim::FormatStringType`
- `ChandleType` maps to `LLVM::LLVMPointerType`
- String could follow similar pattern (LLVM pointer with string metadata)

---

## Environment

- **CIRCT Version**: 1.139.0
- **Dialect**: Moore
- **Failing Pass**: MooreToCore
- **Crash Type**: assertion_failure
- **Crash Hash**: cd336851dcca

---

## Validation Results

### Syntax Check
- **Tool**: slang 10.0.6+3d7e6cd2e
- **Result**: ✅ Valid (0 errors, 0 warnings)
- **Tool**: verilator 5.022
- **Result**: ✅ Valid (2 non-fatal lint warnings)

### Cross-Tool Verification
| Tool | Result | Notes |
|------|--------|-------|
| slang | pass | Valid SystemVerilog |
| verilator | pass_with_warnings | Warnings are non-fatal |
| CIRCT | crash | Assertion failure |

### Classification
- **Result**: `report` (valid bug)
- **Confidence**: high
- **Reason**: Valid SystemVerilog syntax causes CIRCT assertion failure due to missing StringType converter

---

## Suggested Fix Directions

### Option 1: Add StringType Converter
Add a type converter for `StringType` in `MooreToCore::populateTypeConversion()`, possibly mapping to:
- `LLVM::LLVMPointerType` (like ChandleType)
- A simulation-specific type (like FormatStringType → sim::FormatStringType)

### Option 2: Reject with Proper Diagnostic
Add null-type check in `getModulePortInfo()` with proper error emission:
```cpp
auto convertedType = typeConverter.convertType(port.type);
if (!convertedType) {
  emitError(port.loc) << "string-typed ports are not supported in MooreToCore";
  return failure();
}
```

### Option 3: Early Rejection in ImportVerilog
Reject string-typed ports early in the import phase with a clear diagnostic message.

---

## Related Issues

⚠️ **IMPORTANT**: Duplicate check found highly related existing issues:

1. **[#8332](https://github.com/llvm/circt/issues/8332)** - `[MooreToCore] Support for StringType from moore to llvm dialect`
   - Similarity score: **12.5** (high)
   - Directly discusses StringType support in MooreToCore
   - Same root cause: missing StringType converter

2. **[#8283](https://github.com/llvm/circt/issues/8283)** - `[ImportVerilog] Cannot compile forward declared string type`
   - Similarity score: **9.0** (medium-high)
   - Same underlying issue: lack of string type conversion

**Note**: This crash is likely a manifestation of the known issue #8332. The current test case demonstrates a different trigger scenario (string as module output port vs string in other contexts).

---

## Keywords for Search

`string`, `StringType`, `port`, `type conversion`, `MooreToCore`, `dyn_cast`, `non-existent value`, `getModulePortInfo`, `typeConverter`, `convertType`

---

## Full Crash Log

<details>
<summary>Click to expand full stack trace</summary>

```
circt-verilog: /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/Casting.h:650: decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]: Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
Stack dump:
0.	Program arguments: /edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw /home/zhiqing/edazz/eda-vulns/circt-bassertion_cd336851dcca_20260125_194143/origin/bug.sv
 #0 0x00005576dbc6532f llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/lib/Support/Unix/Signals.inc:842:13
 #1 0x00005576dbc662e9 llvm::sys::RunSignalHandlers() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/lib/Support/Signals.cpp:109:18
 #2 0x00005576dbc662e9 SignalHandler(int, siginfo_t*, void*) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/lib/Support/Unix/Signals.inc:412:3
 #3 0x00007f0771361330 (/lib/x86_64-linux-gnu/libc.so.6+0x45330)
 #4 0x00007f07713bab2c __pthread_kill_implementation ./nptl/pthread_kill.c:44:76
 #5 0x00007f07713bab2c __pthread_kill_internal ./nptl/pthread_kill.c:78:10
 #6 0x00007f07713bab2c pthread_kill ./nptl/pthread_kill.c:89:10
 #7 0x00007f077136127e raise ./signal/../sysdeps/posix/raise.c:27:6
 #8 0x00007f07713448ff abort ./stdlib/abort.c:81:7
 #9 0x00007f077134481b _nl_load_domain ./intl/loadmsgcat.c:1177:9
#10 0x00007f0771357517 (/lib/x86_64-linux-gnu/libc.so.6+0x3b517)
#11 0x00005576da3e0874 mlir::TypeStorage::getAbstractType() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/IR/TypeSupport.h:173:5
#12 0x00005576da3e0874 mlir::Type::getTypeID() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/IR/Types.h:101:37
#13 0x00005576da3e0874 bool mlir::detail::StorageUserBase<circt::hw::InOutType, mlir::Type, circt::hw::detail::InOutTypeStorage, mlir::detail::TypeUniquer>::classof<mlir::Type>(mlir::Type) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/IR/StorageUniquerSupport.h:113:16
#14 0x00005576da3e0874 llvm::CastInfo<circt::hw::InOutType, mlir::Type, void>::isPossible(mlir::Type) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/IR/Types.h:374:14
#15 0x00005576da3e0874 llvm::DefaultDoCastIfPossible<circt::hw::InOutType, mlir::Type, llvm::CastInfo<circt::hw::InOutType, mlir::Type, void>>::doCastIfPossible(mlir::Type) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/Casting.h:311:10
#16 0x00005576da3e0874 decltype(auto) llvm::dyn_cast<circt::hw::InOutType, mlir::Type>(mlir::Type&) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/Casting.h:651:10
#17 0x00005576da3e0874 circt::hw::ModulePortInfo::sanitizeInOut() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/include/circt/Dialect/HW/PortImplementation.h:177:24
#18 0x00005576da6d8753 llvm::SmallVectorTemplateCommon<circt::hw::PortInfo, void>::begin() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/ADT/SmallVector.h:271:45
#19 0x00005576da6d8753 llvm::SmallVectorTemplateCommon<circt::hw::PortInfo, void>::end() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/ADT/SmallVector.h:273:27
#20 0x00005576da6d8753 llvm::SmallVector<circt::hw::PortInfo, 1u>::~SmallVector() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/ADT/SmallVector.h:1209:46
#21 0x00005576da6d8753 (anonymous namespace)::getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Conversion/MooreToCore/MooreToCore.cpp:259:1
#22 0x00005576da6d8753 (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, circt::moore::SVModuleOpAdaptor, mlir::ConversionPatternRewriter&) const /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Conversion/MooreToCore/MooreToCore.cpp:276:32
#23 0x00005576da6d8a81 llvm::LogicalResult mlir::ConversionPattern::dispatchTo1To1<mlir::OpConversionPattern<circt::moore::SVModuleOp>, circt::moore::SVModuleOp>(mlir::OpConversionPattern<circt::moore::SVModuleOp> const&, circt::moore::SVModuleOp, circt::moore::SVModuleOp::GenericAdaptor<llvm::ArrayRef<mlir::ValueRange>>, mlir::ConversionPatternRewriter&) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/Transforms/DialectConversion.h:0:15
#24 0x00005576da6d8515 mlir::OpConversionPattern<circt::moore::SVModuleOp>::matchAndRewrite(mlir::Operation*, llvm::ArrayRef<mlir::ValueRange>, mlir::ConversionPatternRewriter&) const /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/Transforms/DialectConversion.h:715:3
#25 0x00005576db6f58ac llvm::SmallVector<mlir::ValueRange, 3u>::~SmallVector() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/ADT/SmallVector.h:0:0
#26 0x00005576db6f58ac mlir::ConversionPattern::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Transforms/Utils/DialectConversion.cpp:2404:1
#27 0x00005576db7541f4 mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>)::$_0::operator()() const /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Rewrite/PatternApplicator.cpp:223:13
#28 0x00005576db7541f4 void llvm::function_ref<void ()>::callback_fn<mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>)::$_0>(long) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/ADT/STLFunctionalExtras.h:46:12
#29 0x00005576db74e6c5 mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Rewrite/PatternApplicator.cpp:242:9
#30 0x00005576db6f674d llvm::succeeded(llvm::LogicalResult) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/LogicalResult.h:0:0
#31 0x00005576db6f674d (anonymous namespace)::OperationLegalizer::legalize(mlir::Operation*) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Transforms/Utils/DialectConversion.cpp:2602:7
#32 0x00005576db6f5a30 llvm::failed(llvm::LogicalResult) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/LogicalResult.h:0:0
#33 0x00005576db6f5a30 mlir::OperationConverter::convert(mlir::Operation*, bool) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Transforms/Utils/DialectConversion.cpp:3300:7
#34 0x00005576db6f6afa llvm::failed(llvm::LogicalResult) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/LogicalResult.h:0:0
#35 0x00005576db6f6afa llvm::LogicalResult mlir::OperationConverter::legalizeOperations<mlir::OperationConverter::applyConversion(llvm::ArrayRef<mlir::Operation*>)::$_0>(llvm::ArrayRef<mlir::Operation*>, mlir::OperationConverter::applyConversion(llvm::ArrayRef<mlir::Operation*>)::$_0, bool) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Transforms/Utils/DialectConversion.cpp:3401:9
#36 0x00005576db6f6afa mlir::OperationConverter::applyConversion(llvm::ArrayRef<mlir::Operation*>) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Transforms/Utils/DialectConversion.cpp:3447:26
#37 0x00005576db70d206 applyConversion(llvm::ArrayRef<mlir::Operation*>, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig, (anonymous namespace)::OpConversionMode)::$_0::operator()() const /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Transforms/Utils/DialectConversion.cpp:4170:9
#38 0x00005576db70d206 void llvm::function_ref<void ()>::callback_fn<applyConversion(llvm::ArrayRef<mlir::Operation*>, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig, (anonymous namespace)::OpConversionMode)::$_0>(long) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/ADT/STLFunctionalExtras.h:46:12
#39 0x00005576db6fd45e llvm::SmallVector<mlir::IRUnit, 6u>::~SmallVector() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/ADT/SmallVector.h:0:0
#40 0x00005576db6fd45e applyConversion(llvm::ArrayRef<mlir::Operation*>, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig, (anonymous namespace)::OpConversionMode) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Transforms/Utils/DialectConversion.cpp:4174:1
#41 0x00005576db6fd5f3 mlir::applyFullConversion(mlir::Operation*, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Transforms/Utils/DialectConversion.cpp:4207:3
#42 0x00005576da6a2832 (anonymous namespace)::MooreToCorePass::runOnOperation() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Conversion/MooreToCore/MooreToCore.cpp:2571:14
#43 0x00005576db8f35a2 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int)::$_3::operator()() const /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Pass/Pass.cpp:0:19
#44 0x00005576db8f35a2 void llvm::function_ref<void ()>::callback_fn<mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int)::$_3>(long) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/ADT/STLFunctionalExtras.h:46:12
#45 0x00005576db8e52b1 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Pass/Pass.cpp:619:17
#46 0x00005576db8e627f llvm::failed(llvm::LogicalResult) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/LogicalResult.h:0:0
#47 0x00005576db8e627f mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Pass/Pass.cpp:688:9
#48 0x00005576db8efce9 mlir::PassManager::runPasses(mlir::Operation*, mlir::AnalysisManager) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Pass/Pass.cpp:1123:3
#49 0x00005576db8eef31 mlir::PassManager::run(mlir::Operation*) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Pass/Pass.cpp:1097:0
#50 0x00005576da333401 executeWithSources(mlir::MLIRContext*, llvm::SourceMgr&) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/tools/circt-verilog/circt-verilog.cpp:398:9
#51 0x00005576da32fed1 execute(mlir::MLIRContext*) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/tools/circt-verilog/circt-verilog.cpp:475:10
#52 0x00005576da32f32f main /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/tools/circt-verilog/circt-verilog.cpp:528:15
#53 0x00007f07713461ca __libc_start_call_main ./csu/../sysdeps/nptl/libc_start_call_main.h:74:3
#54 0x00007f077134628b call_init ./csu/../sysdeps/nptl/libc-start.c:128:20
#55 0x00007f077134628b __libc_start_main ./csu/../sysdeps/nptl/libc-start.c:347:5
#56 0x00005576da32e795 _start (/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog+0x1627795)
```

</details>
