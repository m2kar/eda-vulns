# Validation Report

## Test Case Information
- **Testcase ID**: 260129-0000178d
- **Original File**: source.sv (153 bytes)
- **Minimized File**: bug.sv (41 bytes)
- **Reduction**: 73.2%

## Minimized Test Case
```systemverilog
module top(output string out); endmodule
```

## Reproduction Command
```bash
circt-verilog --ir-hw bug.sv
```

## Validation Results

### 1. Syntax Validity
- **Status**: ✅ VALID
- The test case uses standard SystemVerilog syntax for dynamic string type port declaration.

### 2. Cross-Tool Validation

| Tool | Version | Result | Errors | Warnings |
|------|---------|--------|--------|----------|
| Slang | 10.0.6+3d7e6cd2e | ✅ PASSED | 0 | 0 |
| Verilator | 5.022 | ✅ PASSED | 0 | 0 |

### 3. CIRCT Crash Confirmation
- **Status**: ✅ CRASH CONFIRMED
- **Crash Type**: Assertion failure in MooreToCore conversion pass
- **Crash Location**: `SVModuleOpConversion::matchAndRewrite` in MooreToCore.cpp

### 4. Stack Trace Summary
```
#0 llvm::sys::PrintStackTrace
...
#4 SVModuleOpConversion::matchAndRewrite (MooreToCore.cpp)
#16 MooreToCorePass::runOnOperation (MooreToCore.cpp)
```

## Classification
- **Classification**: `report` (Valid Bug Report)
- **Reason**: The test case uses valid SystemVerilog syntax that is accepted by multiple industry-standard tools (Slang, Verilator), but causes an internal crash in circt-verilog during the Moore-to-Core conversion pass.

## Root Cause Summary
The crash occurs because the `string` type (dynamic string) in SystemVerilog module ports lacks a proper conversion rule in the MooreToCore TypeConverter. When converting `moore::SVModuleOp` to HW module, the TypeConverter returns null for string type ports, which later causes an assertion failure when `sanitizeInOut()` attempts to call `dyn_cast<InOutType>` on a null type.

## Minimality Verification
The test case is minimal because:
1. Single module with single port
2. No body or additional constructs
3. Only the triggering construct (`output string out`) is preserved
4. Removing any part would eliminate the crash trigger

## Conclusion
This is a **valid bug report** for CIRCT. The crash is reproducible and the test case is both syntactically valid and minimal.
