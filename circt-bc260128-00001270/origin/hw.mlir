module {
  hw.module @combined_mod(in %in : i8, in %wide_input : i64, out out_val : i32, in %io_sig : !llhd.ref<i1>, out out : i8) {
    %c0_i7 = hw.constant 0 : i7
    %0 = comb.extract %in from 0 : (i8) -> i1
    %1 = comb.concat %0, %c0_i7 : i1, i7
    %2 = comb.extract %wide_input from 0 : (i64) -> i32
    hw.output %2, %1 : i32, i8
  }
}
