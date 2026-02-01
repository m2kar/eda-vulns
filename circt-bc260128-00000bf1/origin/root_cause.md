# Root Cause Analysis: CIRCT Timeout (260128-00000bf1)

## Summary

**Crash Type**: Timeout (compilation hangs for >300 seconds)  
**Pipeline Stage**: `circt-verilog --ir-hw` (specifically, the LLHD→HW lowering pipeline)  
**Root Cause**: The `llhd-sig2reg` pass generates circular SSA definitions when handling partial struct field assignments in combinational logic, causing the subsequent `canonicalize` pass to enter an infinite loop.

## Testcase Analysis

### Source Code (`source.sv`)

```systemverilog
module test(
  input logic clk
);
  logic [7:0] arr [0:3];          // Unpacked array
  
  typedef struct packed {
    logic valid;
    logic [7:0] data;
  } packet_t;
  
  packet_t pkt;                    // Struct variable
  logic q;
  
  // Partial struct field assignment
  always_comb begin
    pkt.data = arr[0];             // ← Problematic: only assigns 'data' field
  end
  
  // Sequential logic reads different struct field
  always_ff @(posedge clk) begin
    q <= pkt.valid;                // ← Reads 'valid' field
  end
endmodule
```

### Key Constructs

| Construct | Type | Role in Bug |
|-----------|------|-------------|
| `packet_t pkt` | Packed struct | Subject of partial assignment |
| `pkt.data = arr[0]` | Partial field assignment | Triggers circular dependency |
| `q <= pkt.valid` | Struct field read | Uses same struct with different field |
| `always_comb` | Combinational block | Creates continuous assignment semantics |

## Detailed Analysis

### Pipeline Execution Trace

Using `--verbose-pass-executions`, the timeout occurs in this pass sequence:

```
[circt-verilog] Running "hw.module(llhd-mem2reg,llhd-hoist-signals,llhd-deseq,
  llhd-lower-processes,cse,canonicalize,llhd-unroll-loops,cse,canonicalize,
  llhd-remove-control-flow,cse,canonicalize,map-arith-to-comb,llhd-combine-drives,
  llhd-sig2reg,cse,canonicalize,seq-reg-of-vec-to-mem,cse,canonicalize)"
  ← TIMEOUT
```

### Isolation Testing

Through incremental pass testing, the problematic sequence was identified:

| Pass Sequence | Result |
|---------------|--------|
| ...→`llhd-sig2reg` | ✅ Completes |
| ...→`llhd-sig2reg`→`cse` | ✅ Completes |
| ...→`llhd-sig2reg`→`cse`→`canonicalize` | ❌ **TIMEOUT** |

### Malformed IR After `llhd-sig2reg`

After `llhd-sig2reg` + `cse`, the IR contains a **circular SSA definition**:

```mlir
%4 = hw.bitcast %7 : (!hw.struct<valid: i1, data: i8>) -> i9   // Uses %7
%5 = hw.bitcast %4 : (i9) -> !hw.struct<valid: i1, data: i8>   // Uses %4
%7 = hw.struct_inject %5["data"], %6 : !hw.struct<valid: i1, data: i8>  // Uses %5
```

**Dependency Cycle**: `%7 → %4 → %5 → %7`

This violates SSA (Static Single Assignment) form where each value must be defined before use. The `canonicalize` pass attempts to fold/simplify these operations but enters an infinite loop due to the circular dependency.

### Root Cause Mechanism

1. **Input Pattern**: `always_comb` block assigns only one field of a packed struct (`pkt.data = arr[0]`), while another process reads a different field (`pkt.valid`)

2. **LLHD Representation**: The `always_comb` is represented as:
   - `llhd.prb %pkt` - probe current value of struct
   - `hw.struct_inject` - create new struct with modified field
   - `llhd.drv %pkt` - drive the modified struct back

3. **`llhd-sig2reg` Bug**: When converting LLHD signals to HW registers, the pass incorrectly handles the "read-modify-write" pattern:
   - It attempts to create a combinational loop where the struct's output is fed back to create its own input
   - Instead of recognizing this as a synthesis-time constant (struct field `valid` is never assigned), it creates a circular reference

4. **Canonicalize Failure**: The `canonicalize` pass has no guard against circular SSA definitions and enters an infinite loop attempting to simplify the cyclic operations.

## Hypothesis

The `llhd-sig2reg` pass in CIRCT 1.139.0 has a bug in handling partial struct field assignments in combinational logic:

1. When a struct variable has only some fields assigned in `always_comb`, and other fields are read elsewhere
2. The pass generates malformed IR with circular SSA definitions
3. This violates MLIR's SSA invariants and causes downstream passes to hang

The correct behavior should either:
- Error out on undriven struct fields
- Initialize undriven fields to a default value (breaking the cycle)
- Properly lower partial assignments without circular dependencies

## Related CIRCT Components

- **Pass**: `llhd-sig2reg` (lib/Dialect/LLHD/Transforms/Sig2RegPass.cpp)
- **Dialect**: LLHD (LLHD dialect for LLHD IR)
- **Affected Constructs**: `llhd.sig`, `llhd.drv`, `llhd.prb`, `hw.struct_inject`

## Reproduction

```bash
# Generates valid Moore IR
circt-verilog --ir-moore source.sv

# Generates valid LLHD IR
circt-verilog --ir-llhd source.sv

# Times out (>300s)
circt-verilog --ir-hw source.sv

# Minimal reproduction:
circt-verilog --ir-llhd source.sv | circt-opt --pass-pipeline='builtin.module(hw.module(llhd-sig2reg,cse,canonicalize))'
```

## Severity

**High** - Any SystemVerilog code with partial struct field assignments in combinational blocks can trigger this timeout, which is a common coding pattern.
