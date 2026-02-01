# Root Cause Analysis

## Crash Summary
- **Testcase ID**: 260128-00000dd1
- **Crash Type**: assertion failure
- **Location**: `circt::arc::StateType::get()` called from `LowerState.cpp:219`
- **Tool**: arcilator

## Error Context

### Error Message
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
```

### Call Stack
```
#12 circt::arc::StateType::get(mlir::Type)
    → ArcTypes.cpp.inc:108
#13 (anonymous namespace)::ModuleLowering::run()
    → LowerState.cpp:219
#14 (anonymous namespace)::LowerStatePass::runOnOperation()
    → LowerState.cpp:1198
```

### Compilation Command
```bash
circt-verilog --ir-hw test.sv | arcilator | opt -O0 | llc -O0 --filetype=obj -o test.o
```

## Test Case Analysis

### Source Code (`source.sv`)
```systemverilog
module MixPorts(
  input logic clk,
  input logic [63:0] wide_input,
  output logic [31:0] out_val,
  inout logic io_sig           // <-- Key: inout port
);

  logic [31:0] data_array [0:1023];
  
  always @(posedge clk) begin
    out_val <= data_array[wide_input[9:0]];
  end
  
  assign io_sig = data_array[0][0];  // <-- Key: bit-select assignment to inout
endmodule
```

### Key Constructions
1. **Inout port `io_sig`**: Single-bit bidirectional signal
2. **Memory array `data_array`**: 1024-element array of 32-bit words
3. **Bit-select assignment**: `data_array[0][0]` extracts bit 0 from element 0
4. **Continuous assignment to inout**: Creates a driver for the bidirectional port

### Type Flow
1. `data_array[0]` produces a 32-bit value
2. `data_array[0][0]` performs bit-select, producing 1-bit
3. Assignment to `io_sig` (inout) generates LLHD reference type `!llhd.ref<i1>`
4. During arc dialect lowering, this type needs state storage allocation
5. `StateType::get()` is called with `!llhd.ref<i1>` type

## Root Cause

### Primary Cause
The Arc dialect's `StateType` validation requires types with **known bit widths**. The `computeLLVMBitWidth()` function in `lib/Dialect/Arc/ArcTypes.cpp` only handles:
- `seq::ClockType`
- `IntegerType`
- `hw::ArrayType`
- `hw::StructType`

The `!llhd.ref<i1>` type (LLHD reference type) is **not in this list**, causing the verification to fail.

### Detailed Analysis

**In `ArcTypes.cpp:87-92`:**
```cpp
LogicalResult
StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  Type innerType) {
  if (!computeLLVMBitWidth(innerType))
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

**In `LowerState.cpp` around line 219 (within `ModuleLowering::run()`):**
The code allocates storage for module inputs:
```cpp
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(arg.getType()), name, storageArg);
  // ...
}
```

When processing an `inout` port that has been lowered to an LLHD reference type, `StateType::get()` triggers the assertion failure.

### Why LLHD Reference Type Appears
- `inout` ports in SystemVerilog are bidirectional
- CIRCT's LLHD dialect uses `llhd.ref<T>` to model references/pointers
- When `circt-verilog --ir-hw` processes an inout port with bit-select assignment, it produces LLHD reference types
- The Arc dialect's `LowerState` pass doesn't expect or handle these types

## Potential Fix Location

### Option 1: Extend `computeLLVMBitWidth()` (Recommended)
**File**: `lib/Dialect/Arc/ArcTypes.cpp`
**Function**: `computeLLVMBitWidth()`

Add handling for `llhd::RefType`:
```cpp
if (auto refType = dyn_cast<llhd::RefType>(type))
  return computeLLVMBitWidth(refType.getElementType());
```

### Option 2: Handle in `LowerState.cpp`
**File**: `lib/Dialect/Arc/Transforms/LowerState.cpp`
**Location**: Before calling `StateType::get()`

Either:
1. Unwrap LLHD reference types before state allocation
2. Skip/transform inout ports that produce reference types
3. Emit a proper diagnostic instead of assertion

### Option 3: Earlier Pass Handling
Convert LLHD reference types to plain integer types before the `LowerState` pass runs.

## Severity Assessment
- **Impact**: Arcilator crashes on valid SystemVerilog designs with inout ports
- **Workaround**: Avoid inout ports in designs targeted for arcilator simulation
- **Priority**: Medium - affects specific but valid hardware patterns
