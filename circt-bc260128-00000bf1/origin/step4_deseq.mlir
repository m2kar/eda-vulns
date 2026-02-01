module {
  hw.module @test(in %clk : i1) {
    %true = hw.constant true
    %0 = llhd.constant_time <0ns, 0d, 1e>
    %c-1_i2 = hw.constant -1 : i2
    %c0_i9 = hw.constant 0 : i9
    %c0_i32 = hw.constant 0 : i32
    %false = hw.constant false
    %clk_0 = llhd.sig name "clk" %false : i1
    %1 = llhd.prb %clk_0 : i1
    %2 = hw.bitcast %c0_i32 : (i32) -> !hw.array<4xi8>
    %arr = llhd.sig %2 : !hw.array<4xi8>
    %3 = hw.bitcast %c0_i9 : (i9) -> !hw.struct<valid: i1, data: i8>
    %pkt = llhd.sig %3 : !hw.struct<valid: i1, data: i8>
    %q = llhd.sig %false : i1
    %4 = llhd.prb %pkt : !hw.struct<valid: i1, data: i8>
    %5 = llhd.prb %arr : !hw.array<4xi8>
    %6 = llhd.constant_time <0ns, 0d, 1e>
    %7 = llhd.process -> !hw.struct<valid: i1, data: i8> {
      cf.br ^bb2(%4, %5 : !hw.struct<valid: i1, data: i8>, !hw.array<4xi8>)
    ^bb1:  // pred: ^bb2
      cf.br ^bb2(%4, %5 : !hw.struct<valid: i1, data: i8>, !hw.array<4xi8>)
    ^bb2(%12: !hw.struct<valid: i1, data: i8>, %13: !hw.array<4xi8>):  // 2 preds: ^bb0, ^bb1
      %14 = hw.array_get %13[%c-1_i2] : !hw.array<4xi8>, i2
      %15 = hw.struct_inject %12["data"], %14 : !hw.struct<valid: i1, data: i8>
      llhd.wait yield (%15 : !hw.struct<valid: i1, data: i8>), (%4, %5 : !hw.struct<valid: i1, data: i8>, !hw.array<4xi8>), ^bb1
    }
    llhd.drv %pkt, %7 after %6 : !hw.struct<valid: i1, data: i8>
    %8 = llhd.constant_time <0ns, 1d, 0e>
    %9:2 = llhd.combinational -> i1, i1 {
      %true_2 = hw.constant true
      %false_3 = hw.constant false
      %false_4 = hw.constant false
      cf.br ^bb1(%true_2, %false_4, %false_3 : i1, i1, i1)
    ^bb1(%12: i1, %13: i1, %14: i1):  // pred: ^bb0
      %false_5 = hw.constant false
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      %valid = hw.struct_extract %4["valid"] : !hw.struct<valid: i1, data: i8>
      cf.br ^bb3(%true_2, %valid, %true_2 : i1, i1, i1)
    ^bb3(%15: i1, %16: i1, %17: i1):  // pred: ^bb2
      llhd.yield %16, %17 : i1, i1
    }
    %10 = seq.to_clock %1
    %q_1 = seq.firreg %9#0 clock %10 {name = "q"} : i1
    %11 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %q, %q_1 after %11 : i1
    llhd.drv %clk_0, %clk after %0 : i1
    hw.output
  }
}

