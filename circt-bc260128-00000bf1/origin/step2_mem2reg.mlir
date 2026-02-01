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
    llhd.process {
      %6 = llhd.prb %pkt : !hw.struct<valid: i1, data: i8>
      %7 = llhd.prb %arr : !hw.array<4xi8>
      cf.br ^bb2(%6, %7 : !hw.struct<valid: i1, data: i8>, !hw.array<4xi8>)
    ^bb1:  // pred: ^bb2
      %8 = llhd.prb %pkt : !hw.struct<valid: i1, data: i8>
      %9 = llhd.prb %arr : !hw.array<4xi8>
      cf.br ^bb2(%8, %9 : !hw.struct<valid: i1, data: i8>, !hw.array<4xi8>)
    ^bb2(%10: !hw.struct<valid: i1, data: i8>, %11: !hw.array<4xi8>):  // 2 preds: ^bb0, ^bb1
      %12 = hw.array_get %11[%c-1_i2] : !hw.array<4xi8>, i2
      %13 = hw.struct_inject %10["data"], %12 : !hw.struct<valid: i1, data: i8>
      %14 = llhd.constant_time <0ns, 0d, 1e>
      llhd.drv %pkt, %13 after %14 : !hw.struct<valid: i1, data: i8>
      llhd.wait (%4, %5 : !hw.struct<valid: i1, data: i8>, !hw.array<4xi8>), ^bb1
    }
    llhd.process {
      %6 = llhd.prb %clk_0 : i1
      %false_1 = hw.constant false
      %false_2 = hw.constant false
      cf.br ^bb1(%6, %false_1, %false_2 : i1, i1, i1)
    ^bb1(%7: i1, %8: i1, %9: i1):  // 3 preds: ^bb0, ^bb2, ^bb3
      %10 = llhd.constant_time <0ns, 1d, 0e>
      llhd.drv %q, %8 after %10 if %9 : i1
      llhd.wait (%1 : i1), ^bb2(%7 : i1)
    ^bb2(%11: i1):  // pred: ^bb1
      %12 = llhd.prb %clk_0 : i1
      %13 = llhd.prb %pkt : !hw.struct<valid: i1, data: i8>
      %14 = comb.xor bin %11, %true : i1
      %15 = comb.and bin %14, %12 : i1
      %false_3 = hw.constant false
      %false_4 = hw.constant false
      cf.cond_br %15, ^bb3, ^bb1(%12, %false_3, %false_4 : i1, i1, i1)
    ^bb3:  // pred: ^bb2
      %true_5 = hw.constant true
      %valid = hw.struct_extract %13["valid"] : !hw.struct<valid: i1, data: i8>
      cf.br ^bb1(%12, %valid, %true_5 : i1, i1, i1)
    }
    llhd.drv %clk_0, %clk after %0 : i1
    hw.output
  }
}

