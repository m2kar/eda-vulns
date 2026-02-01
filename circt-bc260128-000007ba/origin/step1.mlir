module {
  hw.module @MixPorts(in %clk : i1, in %wide_input : i64, out out_val : i32, in %io_sig : !llhd.ref<i1>) {
    %c32_i32 = hw.constant 32 : i32
    %c1_i32 = hw.constant 1 : i32
    %c0_i32 = hw.constant 0 : i32
    %0 = comb.add %idx, %c1_i32 : i32
    %1 = seq.to_clock %clk
    %idx = seq.firreg %0 clock %1 : i32
    %2 = comb.mods %idx, %c32_i32 : i32
    %3 = comb.concat %c0_i32, %2 : i32, i32
    %4 = comb.shru %wide_input, %3 : i64
    %5 = comb.extract %4 from 0 : (i64) -> i32
    hw.output %5 : i32
  }
}
