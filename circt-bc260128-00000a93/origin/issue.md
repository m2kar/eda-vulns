# [Crash] circt-verilog asserts in MooreToCore when lowering SV `string` port

## Summary
`circt-verilog --ir-hw` crashes with an assertion in
`circt::hw::ModulePortInfo::sanitizeInOut()` while converting a Moore
`SVModuleOp` that contains a `string` input and `string.len()` usage.

## Reproducer

### Testcase (minimized)
```systemverilog
module test(input string a, output int b);
  logic [31:0] shared_signal;
  
  assign shared_signal = a.len();
  assign b = shared_signal;
endmodule
```

### Command
```
/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw bug.sv
```

## Actual Behavior
Assertion failure:
```
dyn_cast on a non-existent value (circt::hw::InOutType)
```

## Expected Behavior
`circt-verilog` should either lower the SV `string` port correctly or emit a
diagnostic explaining that `string` ports are unsupported, without crashing.

## Stack Trace (excerpt)
```
circt::hw::ModulePortInfo::sanitizeInOut()
(anonymous namespace)::getModulePortInfo(...)
SVModuleOpConversion::matchAndRewrite(...)
(anonymous namespace)::MooreToCorePass::runOnOperation()
```

Full logs:
- reproduce.log
- error.log

## Notes / Analysis
- The crash originates in Moore-to-Core conversion while legalizing module ports.
- The `string` input appears to produce an invalid/empty port type; `sanitizeInOut()`
  performs `dyn_cast` without checking that a type is present.

## Environment
- CIRCT binary: `/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog`
- LLVM toolchain: llvm-22 in PATH (as required by local workflow)
- OS: Linux (x86_64)
