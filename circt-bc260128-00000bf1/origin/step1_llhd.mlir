module {
  hw.module @test(in %clk : i1) {
    %0 = llhd.constant_time <0ns, 1d, 0e>
    %true = hw.constant true
    %1 = llhd.constant_time <0ns, 0d, 1e>
    %c-1_i2 = hw.constant -1 : i2
    %c0_i9 = hw.constant 0 : i9
    %c0_i32 = hw.constant 0 : i32
    %false = hw.constant false
    %clk_0 = llhd.sig name "clk" %false : i1
    %2 = llhd.prb %clk_0 : i1
    %3 = hw.bitcast %c0_i32 : (i32) -> !hw.array<4xi8>
    %arr = llhd.sig %3 : !hw.array<4xi8>
    %4 = hw.bitcast %c0_i9 : (i9) -> !hw.struct<valid: i1, data: i8>
    %pkt = llhd.sig %4 : !hw.struct<valid: i1, data: i8>
    %q = llhd.sig %false : i1
    %5 = llhd.prb %pkt : !hw.struct<valid: i1, data: i8>
    %6 = llhd.prb %arr : !hw.array<4xi8>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %7 = llhd.sig.struct_extract %pkt["data"] : <!hw.struct<valid: i1, data: i8>>
      %8 = llhd.prb %arr : !hw.array<4xi8>
      %9 = hw.array_get %8[%c-1_i2] : !hw.array<4xi8>, i2
      llhd.drv %7, %9 after %1 : i8
      llhd.wait (%5, %6 : !hw.struct<valid: i1, data: i8>, !hw.array<4xi8>), ^bb1
    }
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 3 preds: ^bb0, ^bb2, ^bb3
      %7 = llhd.prb %clk_0 : i1
      llhd.wait (%2 : i1), ^bb2
    ^bb2:  // pred: ^bb1
      %8 = llhd.prb %clk_0 : i1
      %9 = comb.xor bin %7, %true : i1
      %10 = comb.and bin %9, %8 : i1
      cf.cond_br %10, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %11 = llhd.prb %pkt : !hw.struct<valid: i1, data: i8>
      %valid = hw.struct_extract %11["valid"] : !hw.struct<valid: i1, data: i8>
      llhd.drv %q, %valid after %0 : i1
      cf.br ^bb1
    }
    llhd.drv %clk_0, %clk after %1 : i1
    hw.output
  }
}
