module {
  hw.module @xor_shift_module(in %clk : i1, out result_out : i1) {
    %0 = hw.aggregate_constant [true, -91 : i8] : !hw.struct<valid: i1, data: i8>
    %c-1_i32 = hw.constant -1 : i32
    %c1_i2 = hw.constant 1 : i2
    %c0_i2 = hw.constant 0 : i2
    %c0_i30 = hw.constant 0 : i30
    %c-1_i2 = hw.constant -1 : i2
    %c3_i32 = hw.constant 3 : i32
    %c4_i32 = hw.constant 4 : i32
    %c1_i32 = hw.constant 1 : i32
    %true = hw.constant true
    %1:6 = llhd.combinational -> !hw.array<4xstruct<valid: i1, data: i8>>, i1, i8, i1, i1, i1 {
      %6 = hw.array_inject %shift_reg[%c-1_i2], %0 : !hw.array<4xstruct<valid: i1, data: i8>>, i2
      cf.br ^bb1(%c1_i32, %6 : i32, !hw.array<4xstruct<valid: i1, data: i8>>)
    ^bb1(%7: i32, %8: !hw.array<4xstruct<valid: i1, data: i8>>):  // 2 preds: ^bb0, ^bb2
      %9 = comb.icmp slt %7, %c4_i32 : i32
      cf.cond_br %9, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %10 = comb.sub %c3_i32, %7 : i32
      %11 = comb.extract %10 from 2 : (i32) -> i30
      %12 = comb.icmp eq %11, %c0_i30 : i30
      %13 = comb.extract %10 from 0 : (i32) -> i2
      %14 = comb.mux %12, %13, %c-1_i2 : i2
      %15 = comb.add %7, %c-1_i32 : i32
      %16 = comb.sub %c3_i32, %15 : i32
      %17 = comb.extract %16 from 2 : (i32) -> i30
      %18 = comb.icmp eq %17, %c0_i30 : i30
      %19 = comb.extract %16 from 0 : (i32) -> i2
      %20 = comb.mux %18, %19, %c-1_i2 : i2
      %21 = hw.array_get %shift_reg[%20] : !hw.array<4xstruct<valid: i1, data: i8>>, i2
      %22 = hw.array_inject %8[%14], %21 : !hw.array<4xstruct<valid: i1, data: i8>>, i2
      %23 = comb.add %7, %c1_i32 : i32
      cf.br ^bb1(%23, %22 : i32, !hw.array<4xstruct<valid: i1, data: i8>>)
    ^bb3:  // pred: ^bb1
      %24 = hw.array_get %shift_reg[%c0_i2] : !hw.array<4xstruct<valid: i1, data: i8>>, i2
      %data = hw.struct_extract %24["data"] : !hw.struct<valid: i1, data: i8>
      %25 = hw.array_get %shift_reg[%c1_i2] : !hw.array<4xstruct<valid: i1, data: i8>>, i2
      %data_0 = hw.struct_extract %25["data"] : !hw.struct<valid: i1, data: i8>
      %26 = comb.xor %data, %data_0 : i8
      %27 = comb.parity %xor_value : i8
      llhd.yield %8, %true, %26, %true, %27, %true : !hw.array<4xstruct<valid: i1, data: i8>>, i1, i8, i1, i1, i1
    }
    %2 = seq.to_clock %clk
    %3 = comb.mux bin %1#1, %1#0, %shift_reg : !hw.array<4xstruct<valid: i1, data: i8>>
    %shift_reg = seq.firreg %3 clock %2 : !hw.array<4xstruct<valid: i1, data: i8>>
    %4 = comb.mux bin %1#3, %1#2, %xor_value : i8
    %xor_value = seq.firreg %4 clock %2 : i8
    %5 = comb.mux bin %1#5, %1#4, %result_out : i1
    %result_out = seq.firreg %5 clock %2 : i1
    hw.output %result_out : i1
  }
}
