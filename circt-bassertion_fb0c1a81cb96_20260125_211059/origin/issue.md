# [MooreToCore] Assertion failure in `sanitizeInOut()` when module has `string` type port

**Bug Type**: Assertion Failure
**Dialect**: Moore
**Failing Pass**: MooreToCore (SVModuleOpConversion)
**Version**: circt-verilog 1.139.0

## Description

CIRCT crashes with an assertion failure when converting a Moore `SVModuleOp` that contains a `string` type output port to core dialects. The crash occurs in `sanitizeInOut()` which attempts to cast port types to `hw::InOutType`, but fails when the port type is `sim::DynamicStringType` (converted from Moore's `StringType`).

### Test Case

```systemverilog
module simple_counter(
  input  logic       clk,
  input  logic       reset,
  output logic [7:0] count,
  output string      status_str
);
  // Counter register
  logic [7:0] count_reg;

  // Sequential logic: counter with synchronous reset
  always_ff @(posedge clk or posedge reset) begin
    if (reset) begin
      count_reg <= 8'b0;
    end else begin
      count_reg <= count_reg + 1;
    end
  end

  // Update output count
  assign count = count_reg;

  // Combinational logic to assign status string based on count value
  always_comb begin
    if (count_reg == 8'b0) begin
      status_str = "RESET";
    end else if (count_reg > 8'd100) begin
      status_str = "OVERFLOW";
    end else begin
      status_str = "COUNTING";
    end
  end
endmodule
```

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv
```

## Error

```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

**Location**: `include/circt/Dialect/HW/PortImplementation.h:177` in `sanitizeInOut()`

### Stack Trace

```
#13 (anonymous namespace)::getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp) MooreToCore.cpp:259
#12 llvm::SmallVector<circt::hw::PortInfo, 1u>::~SmallVector() llvm/ADT/SmallVector.h:1207
#11 (anonymous namespace)::getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp) MooreToCore.cpp:259
```

## Root Cause Analysis

### Issue

The `sanitizeInOut()` function in `ModulePortInfo` constructor (called at line 62 of `PortImplementation.h`) assumes all port types are either:
1. Standard hardware types (which don't match `hw::InOutType` cast), or
2. `hw::InOutType` (which get unwrapped to element type)

However, it doesn't handle types from other dialects like `sim::DynamicStringType`.

### Mechanism

1. **Moore Parsing**: `source.sv` is parsed, creating a `moore.SVModuleOp` with a port of type `moore::StringType`
2. **Type Conversion** (in `MooreToCore.cpp:populateTypeConversion()`, line 2277-2279):
   - `StringType` is converted to `sim::DynamicStringType`
3. **Port Info Extraction** (in `MooreToCore.cpp:getModulePortInfo()`, line 243):
   - Port type is converted using `typeConverter.convertType(port.type)`
   - For `string` port, `portTy` becomes `sim::DynamicStringType`
4. **Sanitization** (in `ModulePortInfo` constructor, line 177 of `PortImplementation.h`):
   - Function iterates through ports and attempts `dyn_cast<hw::InOutType>(p.type)`
   - **FAILS**: `p.type` is `sim::DynamicStringType`, not `hw::InOutType`
   - Assertion triggered: "dyn_cast on a non-existent value"

### Evidence

- Test case has `output string status_str` which is `moore::StringType` ✅
- `StringType` is converted to `sim::DynamicStringType` in MooreToCore.cpp:2278 ✅
- `sanitizeInOut()` performs `dyn_cast<hw::InOutType>(p.type)` without type checking ❌
- Crash occurs at line 177 of PortImplementation.h ❌
- Removing the `string` port from the test case eliminates the crash ✅

## Validation

### Syntax Check
- **Verilator v5.022**: ✅ PASS - No syntax errors
- **Slang v10.0.6**: ✅ PASS - No syntax errors

### IEEE 1800-2005 Compliance
The test case uses standard SystemVerilog constructs:
- `always_ff` - Sequential logic with clock edge (Section 9.4) ✅
- `always_comb` - Combinational logic (Section 9.4) ✅
- `string` type - Dynamic string type (Section 6.16) ✅
- `logic` type - SystemVerilog logic type (Section 6.9) ✅

**Conclusion**: The test case is valid SystemVerilog and uses standard-compliant constructs.

### Classification
**Type**: `genuine_bug` - This is a bug in CIRCT's internal code, not a user error.

### Duplicate Check
- Searched CIRCT issues for: `string port`, `sanitizeInOut`, `InOutType string`, `MooreToCore string`
- Found related issue #8332: "[MooreToCore] Support for StringType from moore to llvm dialect"
  - **Different problem**: Feature request about StringType implementation, not a crash report
- **Similarity score**: 6.0/10.0 (same topic, different issue type)
- **Conclusion**: **Not a duplicate**

## Suggested Fix

Add type checking in `sanitizeInOut()` to guard against non-HW dialect types:

```cpp
// In include/circt/Dialect/HW/PortImplementation.h, lines 175-181
void sanitizeInOut() {
  for (auto &p : ports)
    // Only attempt cast for HW types
    if (!isa<hw::InOutType>(p.type))
      continue;  // Skip non-HW types like sim::DynamicStringType

    if (auto inout = dyn_cast<hw::InOutType>(p.type)) {
      p.type = inout.getElementType();
      p.dir = ModulePort::Direction::InOut;
    }
}
```

Alternatively, explicitly handle `sim::DynamicStringType` if string ports are meant to be supported in the context where `sanitizeInOut()` is called.

## Files

- **Test case**: `bug.sv`
- **Error log**: `error.log`
- **Reproduction command**: `command.txt`
- **Analysis**: `root_cause.md`, `analysis.json`
- **Validation**: `validation.md`, `validation.json`
- **Duplicate check**: `duplicates.md`, `duplicates.json`

## Keywords

`string` `StringType` `DynamicStringType` `sanitizeInOut` `InOutType` `MooreToCore` `SVModuleOp` `ModulePortInfo` `assertion` `crash` `dyn_cast`

## Related Issues

- #8332: "[MooreToCore] Support for StringType from moore to llvm dialect" (related but different issue)
