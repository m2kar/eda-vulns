module {
  hw.module @MixedPorts(in %clk : i1, in %a : i1, out b : i1, in %c : !llhd.ref<i1>) {
    %c0_i3 = hw.constant 0 : i3
    %c0_i2 = hw.constant 0 : i2
    %false = hw.constant false
    %0 = comb.concat %false, %a : i1, i1
    %1 = comb.concat %a, %false : i1, i1
    %2 = comb.or %0, %1 : i2
    %3 = comb.concat %false, %2 : i1, i2
    %4 = comb.concat %a, %c0_i2 : i1, i2
    %5 = comb.or %3, %4 : i3
    %6 = comb.concat %false, %5 : i1, i3
    %7 = comb.concat %a, %c0_i3 : i1, i3
    %8 = comb.or %6, %7 : i4
    %9 = seq.to_clock %clk
    %temp_reg = seq.firreg %8 clock %9 : i4
    %10 = comb.extract %temp_reg from 0 : (i4) -> i1
    hw.output %10 : i1
  }
}
