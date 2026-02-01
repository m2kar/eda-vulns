<!-- 
  CIRCT Bug Report
  Testcase ID: 260129-00001624
-->

<!-- Title: [Moore] Assertion failure in MooreToCore when using packed union as module port -->

## Description

MooreToCore pass crashes when processing a SystemVerilog packed union type used as a module port. The typeConverter.convertType() returns null for moore::UnionType, causing a dyn_cast assertion failure in hw::ModulePortInfo::sanitizeInOut().

**Crash Type**: assertion
**Dialect**: Moore
**Failing Pass**: MooreToCore

**Key Issue**: MooreToCore lacks type converter for `moore::UnionType` (packed union). When converting module ports, the type converter returns null Type, which then triggers an assertion failure when dyn_cast is called on it.

## Steps to Reproduce

1. Save test case below as \`test.sv\`
2. Run:
   \`\`\`bash
   circt-verilog --ir-hw test.sv
   \`\`\`

## Test Case

```systemverilog
typedef union packed { logic [31:0] a; } U;
module top(output U data);
endmodule
```

## Error Output

```
circt-verilog: /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/Casting.h:650: decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]: Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

## Root Cause Analysis

### Missing Type Converter

The MooreToCore pass in CIRCT does not have a type converter registered for \`moore::UnionType\` (packed union). This causes:

1. In \`getModulePortInfo()\`, \`typeConverter.convertType(unionType)\` returns \`Type{}\` (null)
2. The null Type is passed to \`hw::ModulePortInfo::sanitizeInOut()\`
3. \`sanitizeInOut()\` attempts \`dyn_cast<hw::InOutType>(p.type)\`
4. The dyn_cast fails with assertion: "dyn_cast on a non-existent value"

### Crash Location

- **File**: \`include/circt/Dialect/HW/PortImplementation.h:177\`
- **Function**: \`hw::ModulePortInfo::sanitizeInOut\`
- **Assertion**: \`dyn_cast on a non-existent value\`

### Suggested Fix

Add a type converter for \`moore::UnionType\` in \`MooreToCore.cpp\`. Options:
1. Convert packed union to equivalent bit-width integer type (simplest)
2. Convert union members to \`hw::StructType\` fields (preserves structure)
3. Add native \`hw::UnionType\` support (most comprehensive)

## Environment

- **CIRCT Version**: CIRCT 1.139.0
- **OS**: Linux
- **Architecture**: x86_64

## Stack Trace

<details>
<summary>Click to expand stack trace</summary>

\`\`\`
circt-verilog: /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/Casting.h:650: decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]: Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
 #0 0x0000562d123e932f llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/lib/Support/Unix/Signals.inc:842:13
 #1 0x0000562d123ea2e9 llvm::sys::RunSignalHandlers() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/lib/Support/Signals.cpp:109:18
#11 0x0000562d10b64874 mlir::TypeStorage::getAbstractType() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/IR/TypeSupport.h:173:5
#12 0x0000562d10b64874 mlir::Type::getTypeID() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/IR/Types.h:101:37
#13 0x0000562d10b64874 bool mlir::detail::StorageUserBase<circt::hw::InOutType, mlir::Type, circt::hw::detail::InOutTypeStorage, mlir::detail::TypeUniquer>::classof<mlir::Type>(mlir::Type) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/IR/StorageUniquerSupport.h:113:16
#14 0x0000562d10b64874 llvm::CastInfo<circt::hw::InOutType, mlir::Type, void>::isPossible(mlir::Type) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/IR/Types.h:374:14
#15 0x0000562d10b64874 llvm::DefaultDoCastIfPossible<circt::hw::InOutType, mlir::Type, llvm::CastInfo<circt::hw::InOutType, mlir::Type, void>>::doCastIfPossible(mlir::Type) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/Casting.h:311:10
#16 0x0000562d10b64874 decltype(auto) llvm::dyn_cast<circt::hw::InOutType, mlir::Type>(mlir::Type&) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/Casting.h:651:10
#17 0x0000562d10b64874 circt::hw::ModulePortInfo::sanitizeInOut() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/include/circt/Dialect/HW/PortImplementation.h:177:24
#18 0x0000562d10e5c753 llvm::SmallVectorTemplateCommon<circt::hw::PortInfo, void>::begin() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/ADT/SmallVector.h:271:45
#19 0x0000562d10e5c753 llvm::SmallVectorTemplateCommon<circt::hw::PortInfo, void>::end() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/ADT/SmallVector.h:273:27
#20 0x0000562d10e5c753 llvm::SmallVector<circt::hw::PortInfo, 1u>::~SmallVector() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/ADT/SmallVector.h:1209:46
#21 0x0000562d10e5c753 (anonymous namespace)::getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Conversion/MooreToCore/MooreToCore.cpp:259:1
#22 0x0000562d10e5c753 (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, circt::moore::SVModuleOpAdaptor, mlir::ConversionPatternRewriter&) const /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Conversion/MooreToCore/MooreToCore.cpp:276:32
#23 0x0000562d10e5ca81 llvm::LogicalResult mlir::ConversionPattern::dispatchTo1To1<mlir::OpConversionPattern<circt::moore::SVModuleOp>, circt::moore::SVModuleOp>(mlir::OpConversionPattern<circt::moore::SVModuleOp> const&, circt::moore::SVModuleOp, circt::moore::SVModuleOp::GenericAdaptor<llvm::ArrayRef<mlir::ValueRange>>, mlir::ConversionPatternRewriter&) /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/Transforms/DialectConversion.h:0:15
#24 0x0000562d10e5c515 mlir::OpConversionPattern<circt::moore::SVModuleOp>::matchAndRewrite(mlir::Operation*, llvm::ArrayRef<mlir::ValueRange>, mlir::ConversionPatternRewriter&) const /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/Transforms/DialectConversion.h:715:3
#25 0x0000562d11e798ac llvm::SmallVector<mlir::ValueRange, 3u>::~SmallVector() /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/ADT/SmallVector.h:0:0
#26 0x0000562d11e798ac mlir::ConversionPattern::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Transforms/Utils/DialectConversion.cpp:2404:1
#27 0x0000562d11ed81f4 mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>)::$_0::operator()() const /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/Rewrite/PatternApplicator.cpp:223:13
\`\`\`

</details>

## Related Issues

- #8930: [MooreToCore] Crash with sqrt/floor (similar dyn_cast failure in MooreToCore)

Note: This issue is related to #8930 (both involve missing type converters in MooreToCore causing dyn_cast failures), but the triggering constructs are different (packed union vs sqrt/floor).

---
*This issue was generated from testcase ID 260129-00001624 found via fuzzing.*

**Labels**: `Moore`, `crash`, `found-by-fuzzing`
