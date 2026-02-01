module {
  hw.module @counter(in %clk : i1, in %rst : i1, out data_out : i9) {
    %true = hw.constant true
    %c1_i9 = hw.constant 1 : i9
    %c0_i9 = hw.constant 0 : i9
    %0 = comb.extract %data_out from 0 : (i9) -> i1
    %1 = comb.extract %data_out from 1 : (i9) -> i1
    %2 = comb.and %0, %1 : i1
    %3 = comb.xor %2, %true : i1
    %4 = seq.to_clock %clk
    %current_state = seq.firreg %3 clock %4 reset async %rst, %true : i1
    %5 = comb.add %data_out, %c1_i9 : i9
    %data_out = seq.firreg %5 clock %4 reset async %rst, %c0_i9 : i9
    hw.output %data_out : i9
  }
}
