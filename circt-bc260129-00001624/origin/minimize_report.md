# Minimization Report

## Summary
- **Testcase ID**: 260129-00001624
- **Original Size**: 393 bytes (22 lines)
- **Minimized Size**: 81 bytes (3 lines)
- **Reduction**: 79.4%

## Minimized Testcase

```systemverilog
typedef union packed { logic [31:0] a; } U;
module top(output U data);
endmodule
```

## Key Constructs Preserved

Based on root cause analysis, the following essential elements were retained:

1. **`union packed` type definition** - The core trigger for the bug
2. **`output` port with union type** - Required to exercise the faulty code path
3. **Module declaration** - Minimal container for the port

## Removed Elements

The following elements were removed as they are not necessary to trigger the crash:

- Second union member (`logic [31:0] b`)
- Local union variable (`u_data`)
- Assignment statements
- `always_comb` block and conditional logic
- Additional output ports (`q`, `ok`)

## Crash Verification

The minimized testcase produces identical crash signature:

```
Assertion: detail::isPresent(Val) && "dyn_cast on a non-existent value"
Location: llvm/include/llvm/Support/Casting.h:650
Function: llvm::dyn_cast<circt::hw::InOutType, mlir::Type>
```

Stack trace key frames match:
- `circt::hw::ModulePortInfo::sanitizeInOut`
- `getModulePortInfo` 
- `SVModuleOpConversion::matchAndRewrite`
- `MooreToCorePass::runOnOperation`

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv
```
