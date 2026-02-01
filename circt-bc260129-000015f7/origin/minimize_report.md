# Minimization Report

## Overview
- **Testcase ID**: 260129-000015f7
- **Original File**: source.sv (731 bytes, 33 lines)
- **Minimized File**: bug.sv (93 bytes, 5 lines)
- **Reduction**: 87.3%

## Minimization Process

### Step 1: Initial Analysis
Based on analysis.json, identified key constructs triggering the bug:
- Class declaration (`class my_class`)
- Class-type variable declaration (`my_class mc`)
- Class instantiation with `new()` in sequential logic block

### Step 2: Iterative Reduction

#### Iteration 1: Remove module ports and logic
- Removed `rst` port
- Removed combinational logic block (`always @(*)`)
- Removed `computed_value` signal
- Result: Crash preserved

#### Iteration 2: Simplify class definition
- Removed class properties (`logic [7:0] data`)
- Removed class methods (`set_data` function)
- Result: Crash preserved

#### Iteration 3: Simplify sequential block
- Removed if-else structure
- Removed method calls (`mc.set_data()`)
- Kept only `mc = new()` assignment
- Result: Crash preserved

#### Iteration 4: Shorten identifiers
- Renamed `test_module` → `m`
- Renamed `my_class` → `c`
- Renamed `mc` → `o`
- Renamed `clk` → `clk` (kept for clarity)
- Result: Crash preserved

### Step 3: Final Verification
- Verified minimized test case triggers the same error
- Error message: `'hw.bitcast' op result #0 must be Type wherein the bitwidth in hardware is known, but got '!llvm.ptr'`
- Invalid integer bitwidth: `i1073741823` (exceeds MLIR's 16777215-bit limit)

## Minimized Test Case

```systemverilog
module m(input clk);
  class c; endclass
  c o;
  always @(posedge clk) o = new();
endmodule
```

## Essential Constructs

1. **Class declaration**: `class c; endclass` - defines a SystemVerilog class
2. **Class-type variable**: `c o;` - declares a variable of class type
3. **Sequential block**: `always @(posedge clk)` - triggers Mem2Reg pass processing
4. **Class instantiation**: `o = new();` - creates class object dynamically

## Constructs Removed (Non-Essential)

| Construct | Original | Reason for Removal |
|-----------|----------|-------------------|
| Reset input | `input logic rst` | Not needed to trigger bug |
| Combinational block | `always @(*)` | Bug is in sequential processing |
| Class properties | `logic [7:0] data` | Empty class still triggers bug |
| Class methods | `set_data()` function | Method calls not required |
| If-else branching | `if (rst) ... else ...` | Simple assignment suffices |
| Method invocations | `mc.set_data()` | Object creation alone triggers bug |

## Root Cause Confirmation

The bug occurs when:
1. CIRCT processes a class-type variable in sequential logic
2. The Mem2Reg pass attempts to handle the class type
3. The class type's size is incorrectly computed (1073741823 bits)
4. This exceeds MLIR's integer bitwidth limit (16777215 bits)
5. Results in `hw.bitcast` operation error with invalid `!llvm.ptr` type

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv
```
