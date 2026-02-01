#!/bin/bash
cd /root/circt/circt-b5
/opt/firtool/bin/circt-verilog --ir-hw source.sv 2>&1
