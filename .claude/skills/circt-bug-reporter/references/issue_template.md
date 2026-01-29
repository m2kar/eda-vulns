# CIRCT Issue Report Template

Standard format for CIRCT bug reports following project conventions.

## Issue Title Format

```
[Dialect/Tool] Brief description of the crash/error
```

Examples:
- `[Moore] Assertion failed in MooreToCore conversion for union types`
- `[FIRRTL] FullResetAnnotation breaks multi-top modules`
- `[circt-verilog] dyn_cast assertion on non-existent value`

## Issue Body Structure

### 1. Description

One-sentence summary of what's failing.

```markdown
## Description

Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"` failed when processing SystemVerilog code with packed unions.
```

### 2. Steps to Reproduce

Minimal instructions to trigger the bug.

```markdown
## Steps to Reproduce

1. Save the following code as `bug.sv`
2. Run: `circt-verilog --ir-hw bug.sv`
```

### 3. Test Case

Minimal SystemVerilog/FIRRTL code that triggers the bug.

```markdown
## Test Case

```systemverilog
typedef union packed {
  logic [31:0] a;
  logic [31:0] b;
} my_union;

module Top(input my_union in_val);
endmodule
```
```

### 4. Error Output

Key error message (not full stack trace).

```markdown
## Error Output

```
circt-verilog: Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
PLEASE submit a bug report to https://github.com/llvm/circt and include the crash backtrace.
```
```

### 5. Environment

Version information.

```markdown
## Environment

- **CIRCT Version**: firtool-1.139.0
```

### 6. Stack Trace (Collapsible)

Full stack trace in collapsible section.

```markdown
<details>
<summary>Stack Trace</summary>

```
Stack dump:
0.	Program arguments: circt-verilog --ir-hw test.sv
 #0 0x... llvm::sys::PrintStackTrace(...)
 ...
```

</details>
```

## Available Labels

| Label | When to Use |
|-------|-------------|
| `bug` | Always for bug reports |
| `Moore` | Moore dialect (circt-verilog, SystemVerilog) |
| `FIRRTL` | FIRRTL dialect (firtool) |
| `Arc` | Arc dialect (arcilator) |
| `HW` | HW dialect |
| `SV` | SV dialect |
| `Verilog/SystemVerilog` | General Verilog issues |

## Example Complete Issue

```markdown
# [Moore] Assertion failed for packed union module ports

## Description

`dyn_cast` assertion fails when module has packed union type as port.

## Steps to Reproduce

1. Save the following code as `test.sv`
2. Run: `circt-verilog --ir-hw test.sv`

## Test Case

```systemverilog
typedef union packed {
  logic [31:0] a;
  logic [31:0] b;
} my_union;

module Top(input my_union in_val);
endmodule
```

## Error Output

```
circt-verilog: Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
PLEASE submit a bug report to https://github.com/llvm/circt
```

## Environment

- **CIRCT Version**: firtool-1.139.0

<details>
<summary>Stack Trace</summary>

```
Stack dump:
0.	Program arguments: circt-verilog --ir-hw test.sv
 #0 0x000056... llvm::sys::PrintStackTrace
 ...
```

</details>
```
