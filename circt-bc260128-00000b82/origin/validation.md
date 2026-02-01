# Validation Report

## Result
- **Classification**: report
- **Crash reproduced**: yes
- **Signature match**: yes (dyn_cast on a non-existent value at Casting.h:650)

## Minimized Testcase
```
module bug(
  output string s[1:0]
);
endmodule
```

## Reproduction
```
/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw bug.sv
```

## Syntax/Tool Validation
- **Verilator**: `verilator --lint-only bug.sv` (pass, no diagnostics)
- **Slang**: `slang --parse bug.sv` failed (flag unsupported in this build)

## Notes
- The crash occurs during Moore-to-Core port conversion when lowering an unpacked array of `string` ports.
