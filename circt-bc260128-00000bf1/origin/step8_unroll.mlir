module {
  hw.module @test(in %clk : i1) {
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
    %6 = hw.array_get %5[%c-1_i2] : !hw.array<4xi8>, i2
    %7 = hw.struct_inject %4["data"], %6 : !hw.struct<valid: i1, data: i8>
    llhd.drv %pkt, %7 after %0 : !hw.struct<valid: i1, data: i8>
    %valid = hw.struct_extract %4["valid"] : !hw.struct<valid: i1, data: i8>
    %8 = seq.to_clock %1
    %q_1 = seq.firreg %valid clock %8 {name = "q"} : i1
    llhd.drv %q, %q_1 after %0 : i1
    llhd.drv %clk_0, %clk after %0 : i1
    hw.output
  }
}

