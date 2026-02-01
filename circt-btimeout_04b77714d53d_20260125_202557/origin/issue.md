# [Moore] Timeout in MooreToHW during partial struct field assignment in always_comb

## Description

The CIRCT compilation pipeline hangs (non-terminating) when processing a SystemVerilog module where a packed struct has one field assigned within an `always_comb` block and a different field is read via a continuous assignment.

This appears to be a regression or a logic hole in how the Moore dialect lowers procedural struct assignments to the HW dialect. Instead of a crash, the compiler enters an infinite loop or an exponentially slow state, eventually exceeding execution timeouts.

**Crash Type**: Timeout (Non-terminating)
**Affected Dialect**: Moore / HW
**Failing Pass**: MooreToHW conversion

## Steps to Reproduce

1. Save the minimal test case below as `bug.sv`
2. Run the following command:
   ```bash
   circt-verilog --ir-hw bug.sv
   ```
3. Observe that the command does not terminate (timed out after 60s in our testing).

## Test Case

```systemverilog
module M (output logic z);
  struct packed { logic a; logic b; } s;
  always_comb s.a = 1;    // assign to one field
  assign z = s.b;         // read different field
endmodule
```

## Error Output

```
Compilation timed out after 60s
```
(No stack trace is available as the process hangs rather than crashing with an assertion or segfault.)

## Root Cause Analysis

The timeout is triggered by the combination of:
1. A **packed struct** with multiple fields.
2. A **procedural assignment** (`always_comb`) to a *subset* of those fields.
3. A **read access** to a *different* field of the same struct instance.

When `MooreToHW` attempts to lower these operations, it likely enters a state where it repeatedly tries to reconcile the partial procedural assignment with the rest of the struct's bits. This results in either an infinite IR generation loop or a non-terminating transformation pass.

## Cross-Tool Validation

The test case is valid SystemVerilog and passes all major open-source validators:
- **Slang (10.0.6)**: ✅ Clean (0 errors, 0 warnings)
- **Verilator (5.022)**: ✅ Pass (only minor lint warnings: UNUSEDSIGNAL/UNDRIVEN)
- **Icarus Verilog (13.0)**: ✅ Pass (benign sensitivity warning)

## Environment

- **CIRCT Version**: 1.139.0 (and recent development builds)
- **OS**: Linux
- **Architecture**: x86_64

## Related Issues

- #6373: "[Arc] Support hw.wires of aggregate types" (Related but not a duplicate; #6373 focuses on wire support while this issue is specifically about a non-terminating conversion in the Moore frontend).

---
*This issue was generated with assistance from an automated bug reporter.*
