# CIRCT Crash Root Cause Analysis

## Test Case Information
- **Testcase ID**: 260129-000016a2
- **File**: source.sv
- **Issue**: inout port with tri-state logic causing assertion failure in arcilator

## Test Case Description

The test case defines a SystemVerilog module with:
- Two input ports (signed and unsigned 8-bit)
- One output port (8-bit)
- **One inout port (io_port) with tri-state assignment**: `assign io_port = (out_port[0]) ? 1'bz : 1'b0;`

## Crash Details

### Error Message
```
state type must have a known bit width; got '!llhd.ref<i1>'
```

### Stack Trace
```
arcilator: ArcTypes.cpp:108:3: StateType::get()
  -> LowerState.cpp:219:66: ModuleLowering::run()
    -> LowerStatePass::runOnOperation() [line 1198:41]
      -> OpToOpPassAdaptor::run() [Pass.cpp]
```

### Key Components Involved
1. **circt-verilog**: Converts SystemVerilog to MLIR, generating LLHD dialect IR
2. **arcilator**: CIRCT tool for generating circuit simulator code from Arc dialect
3. **LowerState pass**: Converts Arc dialect operations to lower-level forms

## Root Cause Analysis

### Trigger Construct
The crash is triggered by the **inout port with tri-state assignment** in the test case.

### Type Transformation Flow

1. **SystemVerilog Input** (source.sv):
   ```systemverilog
   inout logic io_port
   assign io_port = (out_port[0]) ? 1'bz : 1'b0;
   ```

2. **circt-verilog Transformation**:
   - The tri-state assignment is converted to LLHD dialect
   - Inout ports with multiple drivers create **LLHD reference types** (`!llhd.ref<i1>`)
   - The ref type is a reference to an underlying value, not a primitive type

3. **ModuleLowering::run()** (LowerState.cpp:219):
   ```cpp
   for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
     auto state = RootInputOp::create(allocBuilder, arg.getLoc(),
                                     StateType::get(arg.getType()), name, storageArg);
     allocatedInputs.push_back(state);
   }
   ```

4. **StateType::get()** (ArcTypes.cpp:108):
   - Calls `StateType::verify()` to validate the type
   - `verify()` calls `computeLLVMBitWidth(arg.getType())`

5. **computeLLVMBitWidth()** (ArcTypes.cpp:29-76):
   ```cpp
   static std::optional<uint64_t> computeLLVMBitWidth(Type type) {
     // Handles: seq::ClockType, IntegerType, hw::ArrayType, hw::StructType
     // But NOT: llhd.ref<i1>
     
     if (auto intType = dyn_cast<IntegerType>(type))
       return intType.getWidth();
     // ... other type checks ...
     
     // We don't know anything about any other types.
     return {};
   }
   ```

6. **Assertion Failure**:
   - `computeLLVMBitWidth()` returns `{}` (not found) for `!llhd.ref<i1>`
   - `StateType::verify()` detects missing bit width
   - `StateType::get()` assertion fails with: "state type must have a known bit width"

### Why llhd.ref<i1> Has No Known Bit Width

The `!llhd.ref<i1>` type is a **reference type** that:
- Points to an underlying value (not a primitive bit width itself)
- Is used in LLHD for multi-driver handling and inout ports
- Cannot be directly allocated as a primitive bit storage in Arc's simulator

## Fix Recommendations

### Option 1: Reject Unsupported Types Gracefully
Add type validation in `LowerState.cpp` before calling `StateType::get()`:

```cpp
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto type = arg.getType();
  
  // Reject types that arcilator cannot handle
  if (isa<llhd::RefType>(type)) {
    return emitError() << "arcilator does not support inout ports with "
                       << "tri-state logic (generated llhd.ref types)";
  }
  
  auto state = RootInputOp::create(allocBuilder, arg.getLoc(),
                                   StateType::get(type), name, storageArg);
  allocatedInputs.push_back(state);
}
```

### Option 2: Support llhd.ref Types
Extend `computeLLVMBitWidth()` to handle `llhd::RefType` by:
1. Checking if the reference points to a primitive type (e.g., i1)
2. Returning the bit width of the pointed-to type
3. Ensuring the Arc simulator can represent the reference semantics

### Option 3: Reject inout Ports Entirely
In arcilator's dialect verification or early in the lowering pass, reject modules with inout ports.

## Severity Assessment

- **Severity**: CRASH (Assertion failure, program termination)
- **Impact**: High - arcilator crashes on valid SystemVerilog code with inout ports
- **Confidence**: High - Root cause is clearly identified through stack trace and code analysis
- **Type**: Missing validation / type system mismatch

## Related Components

- **Dialect**: arc (Arc dialect for circuit simulation)
- **Tool**: arcilator (CIRCT circuit simulator generator)
- **Pass**: LowerState (Transforms/LowerState.cpp)
- **Types**: StateType (ArcTypes.cpp)
- **Frontend**: circt-verilog (converts SV to MLIR)

## Reproduction

Run the following command to reproduce:
```bash
cd /home/zhiqing/edazz/eda-vulns/circt-bc260129-000016a2/origin
cat source.sv | circt-verilog --ir-hw - | arcilator | opt -O0
```

Expected: Assertion failure with message "state type must have a known bit width"
