# Validation Report

## Command
```
/opt/firtool/bin/circt-verilog --ir-hw bug.sv | /opt/firtool/bin/arcilator | /opt/llvm-22/bin/opt -O0 | /opt/llvm-22/bin/llc -O0 --filetype=obj -o /tmp/circt_260129_0000129c_min.o
```

## Result
Reproduced the same legalization failure on `sim.fmt.literal` with the minimized
test case.

## Observed Error
```
<stdin>:3:10: error: failed to legalize operation 'sim.fmt.literal'
    %0 = sim.fmt.literal "Error: Assertion failed: q != 0"
         ^
```
