# CIRCT Bug #260129-0000159f - Documentation Index

## Quick Navigation

### 📖 Read This First
1. **QUICKREF.txt** - Quick reference guide with test cases and key info
2. **minimize.md** - Detailed minimization analysis and process

### 📊 For Data Analysis
- **minimize.json** - Machine-readable structured data
- **MINIMIZATION_SUMMARY.txt** - Task completion summary

### 🔧 Original Files
- **source.sv** - Original test case (10 lines, UNCHANGED)
- **error.txt** - Original crash error log
- **crash_signature_analysis.txt** - Crash analysis details

## File Descriptions

| File | Type | Size | Purpose |
|------|------|------|---------|
| source.sv | SystemVerilog | 288 bytes | Original test case causing tri-state assertion failure |
| minimize.md | Markdown | 4.6 KB | Comprehensive minimization report with analysis |
| minimize.json | JSON | 4.3 KB | Structured data - all test variants and analysis |
| MINIMIZATION_SUMMARY.txt | Text | 4.9 KB | Task completion checklist and recommendations |
| QUICKREF.txt | Text | 2.0 KB | Quick reference for developers |
| error.txt | Text | 1.4 KB | Original crash error message and stack trace |
| crash_signature_analysis.txt | Text | 3.0 KB | Detailed crash signature extraction |

## Key Information

### The Bug
- **Type:** Tri-state Inout Port - Assertion Failure
- **Error:** `state type must have a known bit width; got '!llhd.ref<i1>'`
- **Location:** `circt::arc::StateType::get(mlir::Type)`
- **Status:** ✅ FIXED in CIRCT 1.139.0+

### Test Cases

**Original (10 lines):**
```verilog
module example(input logic clk, inout logic c);
  logic [3:0] temp_reg;
  logic a;
  always @(posedge clk) begin
    temp_reg <= temp_reg + 1;
  end
  assign c = (a) ? temp_reg[0] : 1'bz;
endmodule
```

**Minimal (4 lines - 60% reduction):**
```verilog
module example(input logic clk, inout logic c);
  logic a;
  assign c = a ? 1'b1 : 1'bz;
endmodule
```

## Compilation Command

```bash
/opt/firtool/bin/circt-verilog --ir-hw source.sv | \
/opt/firtool/bin/arcilator
```

## Critical Elements

**Must Include:**
- `inout logic` port with tri-state capability
- `1'bz` (high-impedance) value
- Conditional ternary operator selecting between value and tri-state

**Can Remove:**
- Sequential `always @(posedge clk)` logic
- Register variables
- Complex array indexing

## Verification Status

✅ All test case variants compile successfully in current toolchain
✅ Original crash cannot be reproduced
✅ Bug has been fixed upstream in CIRCT Arc dialect

## Related Source Code

Monitor these files in the CIRCT repository:
- `lib/Dialect/Arc/ArcTypes.cpp`
- `lib/Dialect/Arc/Transforms/LowerState.cpp`
- `lib/Dialect/LLHD/IR/LLHDTypes.cpp`

Key functions to review:
- `circt::arc::StateType::get()`
- `circt::arc::detail::StateTypeStorage`
- `ModuleLowering::run()`
- `LowerStatePass::runOnOperation()`

## Document Versions

- **minimize.md**: Version 1.0 - Initial analysis
- **minimize.json**: Version 1.0 - Initial structured data
- **QUICKREF.txt**: Version 1.0 - Quick reference
- **INDEX.md**: Version 1.0 - This index

## Next Steps

1. **For Regression Testing:** Use the 4-line minimal variant
2. **For Documentation:** Reference the detailed minimize.md report
3. **For Bug Tracking:** Check minimize.json for structured data
4. **For Quick Info:** See QUICKREF.txt

---
*Analysis completed: 2025-02-01*
*Toolchain: CIRCT 1.139.0+, LLVM 22.0.0git*
*Platform: Linux x86_64*
