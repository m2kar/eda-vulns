# IEEE 1800-2005 SystemVerilog Quick Reference

Quick reference for common SystemVerilog constructs relevant to CIRCT bug validation.

## Data Types (Section 4-7)

### Logic Types
| Type | Description | Section |
|------|-------------|---------|
| `logic` | 4-state type (0, 1, X, Z) | 4.3 |
| `reg` | Variable, 4-state | 4.3 |
| `wire` | Net, 4-state | 4.3 |
| `bit` | 2-state type (0, 1) | 4.3 |
| `integer` | 32-bit signed 4-state | 4.4 |
| `real` | 64-bit floating point | 4.5 |

### Packed Arrays (Section 7.4)
```systemverilog
logic [7:0] byte_var;           // packed array
logic [3:0][7:0] packed_2d;     // packed multi-dimensional
```

### Unpacked Arrays (Section 7.4)
```systemverilog
logic [7:0] mem [0:255];        // unpacked array (memory)
logic arr [3][4];               // unpacked multi-dimensional
```

### Dynamic Arrays (Section 7.5) - **Often Unsupported in Synthesis**
```systemverilog
logic [7:0] dyn_arr[];          // dynamic array
```

### Queues (Section 7.10) - **Often Unsupported in Synthesis**
```systemverilog
logic [7:0] queue[$];           // queue
logic [7:0] bounded[$:100];     // bounded queue
```

### Associative Arrays (Section 7.8) - **Often Unsupported in Synthesis**
```systemverilog
logic [31:0] assoc[*];          // associative array (wildcard)
logic [31:0] assoc[string];     // associative array (string key)
```

## Structures and Unions (Section 7.2)

### Packed Struct
```systemverilog
typedef struct packed {
    logic [7:0] field1;
    logic [7:0] field2;
} my_struct_t;
```

### Packed Union
```systemverilog
typedef union packed {
    logic [15:0] word;
    logic [1:0][7:0] bytes;
} my_union_t;
```

### Nested Types - **May Have Limited Support**
```systemverilog
typedef struct packed {
    union packed {
        logic [31:0] a;
        logic [3:0][7:0] b;
    } inner;
} nested_t;
```

## Procedural Blocks (Section 9)

### Always Blocks
```systemverilog
always_ff @(posedge clk)        // sequential, single clock edge
always_comb                      // combinational, implicit sensitivity
always_latch                     // latch inference
always @(posedge clk or negedge rst)  // async reset
```

### Sensitivity Lists (Section 9.2.2)
```systemverilog
always @(a or b)                // OR sensitivity
always @(*)                     // implicit sensitivity
always @(posedge clk)           // edge sensitivity
always @(arr[0])                // array element - may have issues
```

**Note**: Array element indexing in sensitivity lists (e.g., `@(arr[0])`) is valid IEEE 1800 syntax but may not be fully supported in all tools.

## Modules and Ports (Section 23)

### Port Declarations
```systemverilog
module Top(
    input  logic        clk,
    input  logic [7:0]  data_in,
    output logic [7:0]  data_out,
    inout  wire  [3:0]  bidir
);
```

### Interface Ports (Section 25)
```systemverilog
interface my_if;
    logic [7:0] data;
    modport master (output data);
    modport slave  (input  data);
endinterface
```

## Operators (Section 11)

### Reduction Operators
```systemverilog
&a    // AND reduction
|a    // OR reduction
^a    // XOR reduction
~&a   // NAND reduction
~|a   // NOR reduction
~^a   // XNOR reduction
```

### Part-Select
```systemverilog
a[7:0]        // fixed part-select
a[base+:8]    // ascending part-select
a[base-:8]    // descending part-select
```

## Generate Constructs (Section 27)

```systemverilog
generate
    for (genvar i = 0; i < 4; i++) begin : gen_block
        // generated instances
    end
endgenerate
```

## Assertions (Section 16-17) - **Often Verification-Only**

### Immediate Assertions
```systemverilog
assert (a == b) else $error("Mismatch");
```

### Concurrent Assertions - **Usually Not Synthesizable**
```systemverilog
assert property (@(posedge clk) a |-> b);
```

## Classes (Section 8) - **Not Synthesizable**
```systemverilog
class MyClass;
    int data;
    function new();
        data = 0;
    endfunction
endclass
```

## Synthesis vs Verification Constructs

### Synthesizable
- Module definitions
- Always blocks (always_ff, always_comb, always_latch)
- Continuous assignments
- Packed arrays and structs
- Generate blocks
- Basic operators

### Usually Not Synthesizable (Verification Only)
- Classes
- Randomization (rand, randc, randomize)
- Covergroups
- Program blocks
- Concurrent assertions
- Dynamic arrays
- Queues
- Associative arrays
- DPI imports

## Common CIRCT Limitations

| Feature | CIRCT Support | Notes |
|---------|---------------|-------|
| Packed structs | ✅ Supported | Full support |
| Packed unions | ⚠️ Partial | May have edge cases |
| Interfaces | ⚠️ Partial | Basic support |
| Dynamic arrays | ❌ Not supported | Use fixed arrays |
| Queues | ❌ Not supported | Verification only |
| Classes | ❌ Not supported | Not synthesizable |
| DPI | ❌ Not supported | Use alternatives |
| Array in sensitivity | ⚠️ Known issues | See GitHub issues |

## References

- IEEE Std 1800-2005: SystemVerilog Standard
- IEEE Std 1800-2012: Updated Standard
- CIRCT Documentation: https://circt.llvm.org/
- Moore Dialect: https://circt.llvm.org/docs/Dialects/Moore/
