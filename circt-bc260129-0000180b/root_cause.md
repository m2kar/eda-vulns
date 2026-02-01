# Root Cause Analysis Report

## Executive Summary
CIRCT compiler crashes when processing SystemVerilog code with a self-referential typedef in a parameterized class. The issue manifests in two ways depending on the version: as an assertion failure `numBits >= 0` in the HoistSignals pass (original crash), or as a semantic error about invalid bitcast with an excessive bit width (current version). Both symptoms point to the same underlying problem: failure to handle recursive type definitions properly, leading to incorrect bit width calculation or overflow.

## Crash Context
- **Tool/Command**: circt-verilog (part of CIRCT toolchain)
- **Dialect**: Moore (SystemVerilog) → LLHD → HW dialect
- **Failing Pass**: HoistSignals pass in LLHD dialect (original), or earlier in Moore→HW conversion (current)
- **Crash Type**: Assertion failure (original), Semantic error (current)

## Error Analysis

### Original Crash (Version 1.139.0)
```
Assertion `numBits >= 0' failed at HoistSignals.cpp:510
Stack trace shows crash in DriveHoister::hoistDrives()
```

**Location**: `/lib/Dialect/LLHD/Transforms/HoistSignals.cpp:510`

```cpp
auto numBits = hw::getBitWidth(type);
assert(numBits >= 0);  // Line 549 - assertion fails here
```

### Current Error (Version 1.139.0, current build)
```
error: 'hw.bitcast' op result #0 must be Type wherein the bitwidth in hardware is known, but got '!llvm.ptr'
note: see current operation: %20 = "hw.bitcast"(%19) : (i1073741823) -> !llvm.ptr
```

**Observations**:
- The bit width `i1073741823` (approximately 2^30) is clearly invalid
- This suggests overflow or infinite recursion in type size calculation
- The error occurs earlier in the pipeline than the original assertion

### Key Stack Frames (Original)
- `HoistSignals.cpp:510` - Assertion failure in DriveHoister
- `ProcessOp::create` - LLHD process creation
- `OpToOpPassAdaptor::runPipeline` - MLIR pass pipeline execution

## Test Case Analysis

### Code Summary
The test case demonstrates SystemVerilog's parameterized classes and typedefs:
1. A package `pkg` containing a parameterized class template `container#(type T = int)`
2. A class `my_class` with a self-referential typedef: `typedef container#(pkg::my_class) my_type`
3. A module `System` that instantiates `my_class` and uses its typedef

### Key Constructs
- **Parameterized class**: `container#(type T = int)` - generic class template
- **Self-referential typedef**: `typedef container#(pkg::my_class) my_type` - creates circular dependency
- **Package import**: `import pkg::*` - brings package contents into scope
- **Class instantiation**: `class_obj = new()` - creates instance in initial block

### Potentially Problematic Patterns
1. **Self-referential typedef**: `typedef container#(pkg::my_class) my_type`
   - Creates a circular type reference: my_type → container#(my_class) → my_type (via pkg::)
   - This type is recursively defined with no base case

2. **Parameterized class with self-reference**: The type `container#(pkg::my_class)` where the parameter refers back to the containing class creates infinite recursion in type resolution

3. **Size calculation**: When the compiler attempts to calculate the bit width of this type, it likely:
   - Recursively expands the typedef
   - Never reaches a terminal type with known size
   - Either crashes on assertion (if returns -1) or overflows (if keeps accumulating)

## CIRCT Source Analysis

### Crash Location (Original)
**File**: `lib/Dialect/LLHD/Transforms/HoistSignals.cpp`
**Function**: `DriveHoister::hoistDrives()`
**Lines**: 509-557

### Code Context
```cpp
void DriveHoister::hoistDrives() {
  if (driveSets.empty())
    return;

  auto materialize = [&](DriveValue driveValue) -> Value {
    OpBuilder builder(processOp);
    return TypeSwitch<DriveValue::Data, Value>(driveValue.data)
        .Case<Value>([](auto value) { return value; })
        .Case<IntegerAttr>([&](auto attr) {
          auto &slot = materializedConstants[attr];
          if (!slot)
            slot = hw::ConstantOp::create(builder, processOp.getLoc(), attr);
          return slot;
        })
        .Case<TimeAttr>([&](auto attr) {
          auto &slot = materializedConstants[attr];
          if (!slot)
            slot = ConstantTimeOp::create(builder, processOp.getLoc(), attr);
          return slot;
        })
        .Case<Type>([&](auto type) {
          // TODO: This should probably create something like a `llhd.dontcare`.
          if (isa<TimeType>(type)) {
            auto attr = TimeAttr::get(builder.getContext(), 0, "ns", 0, 0);
            auto &slot = materializedConstants[attr];
            if (!slot)
              slot = ConstantTimeOp::create(builder, processOp.getLoc(), attr);
            return slot;
          }
          auto numBits = hw::getBitWidth(type);  // Line 548
          assert(numBits >= 0);                    // Line 549 - ASSERTION FAILS HERE
          Value value = hw::ConstantOp::create(
              builder, processOp.getLoc(), builder.getIntegerType(numBits), 0);
          if (value.getType() != type)
              value =
                  hw::BitcastOp::create(builder, processOp.getLoc(), type, value);
          return value;
        });
  };
  // ... rest of function
}
```

### getBitWidth Implementation
**File**: `lib/Dialect/HW/HWTypes.cpp`

```cpp
int64_t circt::hw::getBitWidth(mlir::Type type) {
  return llvm::TypeSwitch<::mlir::Type, int64_t>(type)
      .Case<IntegerType>(
          [](IntegerType t) { return t.getIntOrFloatBitWidth(); })
      .Default([](Type type) -> int64_t {
        // If type implements the BitWidthTypeInterface, use it.
        if (auto iface = dyn_cast<BitWidthTypeInterface>(type)) {
          std::optional<int64_t> width = iface.getBitWidth();
          return width.has_value() ? *width : -1;  // Returns -1 on error
        }
        return -1;  // Returns -1 for types without the interface
      });
}
```

**Key Finding**: The function returns `-1` when:
1. The type doesn't implement `BitWidthTypeInterface`
2. The interface's `getBitWidth()` returns `std::nullopt`
3. This triggers the assertion failure in HoistSignals.cpp

### StructType::getBitWidth Implementation
```cpp
std::optional<int64_t> StructType::getBitWidth() const {
  int64_t total = 0;
  for (auto field : getElements()) {
    int64_t fieldSize = hw::getBitWidth(field.type);
    if (fieldSize < 0)
      return std::nullopt;  // Propagates error up
    total += fieldSize;
  }
  return total;
}
```

## Processing Path

1. **Moore Parser**: Parses SystemVerilog source, creates MLIR IR with Moore dialect
   - Encounters parameterized class `container#(pkg::my_class)`
   - Creates type alias for `typedef container#(pkg::my_class) my_type`

2. **Moore to HW Conversion**: Lowers Moore dialect to HW dialect
   - Attempts to calculate bit width for all types
   - Encounters self-referential typedef
   - Recursively expands `my_type` → `container#(pkg::my_class)` → `my_type` → ...
   - Either:
     - Detects the cycle and returns `-1` (causing assertion), OR
     - Keeps accumulating sizes until overflow (producing `i1073741823`)

3. **HoistSignals Pass** (if reached):
   - Needs to materialize drive values with default constants
   - Calls `hw::getBitWidth(type)` to create a constant
   - Receives `-1` or overflow value
   - Assertion fails (original crash) OR validation error occurs (current version)

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence): Missing Circular Type Detection
**Cause**: The Moore dialect's type system does not properly detect or handle self-referential typedefs, leading to infinite recursion during bit width calculation.

**Evidence**:
- The self-referential typedef `typedef container#(pkg::my_class) my_type` creates a circular dependency
- `hw::getBitWidth()` returns `-1` when unable to calculate width, triggering assertion
- Current version shows overflow value `i1073741823`, suggesting unbounded recursion before detection
- StructType implementation shows error propagation logic, but Moore dialect may not use it

**Mechanism**:
1. Type resolution attempts to expand `container#(pkg::my_class)`
2. Finds `my_class` with typedef to `container#(pkg::my_class)`
3. Recursively follows the typedef without cycle detection
4. Either detects error late (returning -1) or overflows (accumulating bits)
5. Passes invalid type to `hw::getBitWidth()` which fails assertion or validation

### Hypothesis 2 (Medium Confidence): Parameterized Class Size Calculation Bug
**Cause**: The logic for calculating size of parameterized classes with recursive type parameters is flawed.

**Evidence**:
- The type involves a parameterized class `container#(type T = int)`
- When `T = pkg::my_class`, it creates recursion
- The size calculation may add class size repeatedly instead of using pointer indirection

**Mechanism**:
1. In SystemVerilog, a class instance is typically a reference/pointer
2. However, the compiler may try to inline the full type size
3. When parameterized with itself, this creates exponential or unbounded growth
4. No safeguard limits this expansion

### Hypothesis 3 (Low Confidence): HW Dialect Assertion Too Strict
**Cause**: The assertion `assert(numBits >= 0)` in HoistSignals is too strict for types with unknown or invalid bit widths.

**Evidence**:
- The assertion fails when `getBitWidth()` returns `-1` (error indicator)
- Alternative could be to handle the error gracefully with a diagnostic

**Mechanism**:
1. Type system correctly identifies problematic type
2. Returns `-1` as error indicator
3. HoistSignals could emit diagnostic instead of asserting
4. This would be a symptom handling issue, not root cause

**Why Lower Confidence**: Even with better error handling, the underlying type calculation bug would still exist.

## Suggested Fix Directions

### 1. Add Circular Type Detection (Recommended)
- Implement cycle detection in Moore dialect's type resolution
- Detect self-referential typedefs and parameterized types
- Either:
  - Reject such constructs with a clear diagnostic, OR
  - Use pointer/reference semantics for recursive types

**Implementation Points**:
- Add visited set to type resolution in `MooreTypes.cpp`
- Track type expansion during bit width calculation
- Emit diagnostic: "self-referential typedef in parameterized class not supported"

### 2. Improve Error Handling in HoistSignals
- Replace assertion `assert(numBits >= 0)` with proper error handling
- Emit diagnostic: "cannot materialize drive value for type with unknown bit width"
- This prevents crash and provides actionable feedback

**File**: `lib/Dialect/LLHD/Transforms/HoistSignals.cpp:549`

### 3. Limit Type Size Calculation
- Add bounds checking to bit width calculation
- Emit diagnostic when size exceeds reasonable limit
- Prevents overflow issues like `i1073741823`

**Implementation Points**:
- Add maximum bit width constant (e.g., 2^16 or 2^20)
- Check during size accumulation in `StructType::getBitWidth()`
- Emit diagnostic with actual and maximum sizes

### 4. Document Unsupported Features
- Add remark/warning for unsupported class features
- Currently shows: "Class builtin functions...not yet supported"
- Should also note: "Self-referential typedefs in parameterized classes not supported"

## Keywords for Issue Search
`self-referential typedef` `circular type` `recursive type` `parameterized class` `HoistSignals` `numBits` `getBitWidth` `container class` `type alias` `Moore dialect`

## Related Files to Investigate
- `lib/Dialect/Moore/MooreTypes.cpp` - Type resolution and bit width calculation
- `lib/Dialect/LLHD/Transforms/HoistSignals.cpp:549` - Assertion location
- `lib/Dialect/HW/HWTypes.cpp` - getBitWidth implementation
- `include/circt/Dialect/HW/HWTypes.h` - BitWidthTypeInterface
- `lib/Conversion/MooreToCore/` - Moore to HW conversion logic

## Additional Observations

### Version Difference
The original crash (assertion) doesn't reproduce in the current version, which now shows a semantic error. This suggests:
1. A validation was added earlier in the pipeline
2. The overflow is now caught before reaching HoistSignals
3. This is actually an improvement in error reporting, but the root cause remains

### Current Error Value
The bit width `1073741823` = 2^30 - 1, which is suspiciously close to a signed 32-bit integer maximum. This suggests:
1. The calculation may be using signed 32-bit arithmetic somewhere
2. The size accumulator overflowed or hit a boundary
3. This could be an intermediate calculation bug, not just infinite recursion

### Test Case Minimization Potential
The test case can likely be minimized to just:
```systemverilog
package pkg;
  class my_class;
    typedef pkg::my_class self_type;
  endclass
endpackage
```

Or even simpler, focusing on the self-referential aspect without modules or initial blocks.
