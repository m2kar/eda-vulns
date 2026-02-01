module {
  hw.module @MixedPorts(in %clk : i1, in %a : i16, in %b : i16, out out_b : i1, in %c : !llhd.ref<i1>) {
    %0 = comb.extract %a from 0 : (i16) -> i1
    %1 = comb.extract %b from 0 : (i16) -> i1
    %2 = comb.add %0, %1 : i1
    %3 = seq.to_clock %clk
    %out_b = seq.firreg %2 clock %3 : i1
    hw.output %out_b : i1
  }
}
