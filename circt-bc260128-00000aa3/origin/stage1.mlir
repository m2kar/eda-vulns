module {
  hw.module @array_reg(in %clk : i1, in %data_in : i8, out sum_out : i12) {
    %c1_i4 = hw.constant 1 : i4
    %c2_i4 = hw.constant 2 : i4
    %c3_i4 = hw.constant 3 : i4
    %c4_i4 = hw.constant 4 : i4
    %c5_i4 = hw.constant 5 : i4
    %c6_i4 = hw.constant 6 : i4
    %c7_i4 = hw.constant 7 : i4
    %c-8_i4 = hw.constant -8 : i4
    %c-7_i4 = hw.constant -7 : i4
    %c-6_i4 = hw.constant -6 : i4
    %c-5_i4 = hw.constant -5 : i4
    %c-4_i4 = hw.constant -4 : i4
    %c-3_i4 = hw.constant -3 : i4
    %c-2_i4 = hw.constant -2 : i4
    %true = hw.constant true
    %c-1_i32 = hw.constant -1 : i32
    %c0_i4 = hw.constant 0 : i4
    %c-1_i4 = hw.constant -1 : i4
    %c0_i28 = hw.constant 0 : i28
    %c1_i32 = hw.constant 1 : i32
    %c15_i32 = hw.constant 15 : i32
    %c16_i32 = hw.constant 16 : i32
    %0 = hw.array_get %arr[%c-1_i4] : !hw.array<16xi8>, i4
    %1 = comb.concat %c0_i4, %0 : i4, i8
    %2 = hw.array_get %arr[%c-2_i4] : !hw.array<16xi8>, i4
    %3 = comb.concat %c0_i4, %2 : i4, i8
    %4 = hw.array_get %arr[%c-3_i4] : !hw.array<16xi8>, i4
    %5 = comb.concat %c0_i4, %4 : i4, i8
    %6 = hw.array_get %arr[%c-4_i4] : !hw.array<16xi8>, i4
    %7 = comb.concat %c0_i4, %6 : i4, i8
    %8 = hw.array_get %arr[%c-5_i4] : !hw.array<16xi8>, i4
    %9 = comb.concat %c0_i4, %8 : i4, i8
    %10 = hw.array_get %arr[%c-6_i4] : !hw.array<16xi8>, i4
    %11 = comb.concat %c0_i4, %10 : i4, i8
    %12 = hw.array_get %arr[%c-7_i4] : !hw.array<16xi8>, i4
    %13 = comb.concat %c0_i4, %12 : i4, i8
    %14 = hw.array_get %arr[%c-8_i4] : !hw.array<16xi8>, i4
    %15 = comb.concat %c0_i4, %14 : i4, i8
    %16 = hw.array_get %arr[%c7_i4] : !hw.array<16xi8>, i4
    %17 = comb.concat %c0_i4, %16 : i4, i8
    %18 = hw.array_get %arr[%c6_i4] : !hw.array<16xi8>, i4
    %19 = comb.concat %c0_i4, %18 : i4, i8
    %20 = hw.array_get %arr[%c5_i4] : !hw.array<16xi8>, i4
    %21 = comb.concat %c0_i4, %20 : i4, i8
    %22 = hw.array_get %arr[%c4_i4] : !hw.array<16xi8>, i4
    %23 = comb.concat %c0_i4, %22 : i4, i8
    %24 = hw.array_get %arr[%c3_i4] : !hw.array<16xi8>, i4
    %25 = comb.concat %c0_i4, %24 : i4, i8
    %26 = hw.array_get %arr[%c2_i4] : !hw.array<16xi8>, i4
    %27 = comb.concat %c0_i4, %26 : i4, i8
    %28 = hw.array_get %arr[%c1_i4] : !hw.array<16xi8>, i4
    %29 = comb.concat %c0_i4, %28 : i4, i8
    %30 = hw.array_get %arr[%c0_i4] : !hw.array<16xi8>, i4
    %31 = comb.concat %c0_i4, %30 : i4, i8
    %32 = comb.add %1, %3, %5, %7, %9, %11, %13, %15, %17, %19, %21, %23, %25, %27, %29, %31 : i12
    %33 = seq.to_clock %clk
    %sum_out = seq.firreg %32 clock %33 : i12
    %34:2 = llhd.combinational -> !hw.array<16xi8>, i1 {
      %36 = hw.array_inject %arr[%c-1_i4], %data_in : !hw.array<16xi8>, i4
      cf.br ^bb1(%c1_i32, %36 : i32, !hw.array<16xi8>)
    ^bb1(%37: i32, %38: !hw.array<16xi8>):  // 2 preds: ^bb0, ^bb2
      %39 = comb.icmp slt %37, %c16_i32 : i32
      cf.cond_br %39, ^bb2, ^bb3(%38, %true : !hw.array<16xi8>, i1)
    ^bb2:  // pred: ^bb1
      %40 = comb.sub %c15_i32, %37 : i32
      %41 = comb.extract %40 from 4 : (i32) -> i28
      %42 = comb.icmp eq %41, %c0_i28 : i28
      %43 = comb.extract %40 from 0 : (i32) -> i4
      %44 = comb.mux %42, %43, %c-1_i4 : i4
      %45 = comb.add %37, %c-1_i32 : i32
      %46 = comb.sub %c15_i32, %45 : i32
      %47 = comb.extract %46 from 4 : (i32) -> i28
      %48 = comb.icmp eq %47, %c0_i28 : i28
      %49 = comb.extract %46 from 0 : (i32) -> i4
      %50 = comb.mux %48, %49, %c-1_i4 : i4
      %51 = hw.array_get %arr[%50] : !hw.array<16xi8>, i4
      %52 = hw.array_inject %38[%44], %51 : !hw.array<16xi8>, i4
      %53 = comb.add %37, %c1_i32 : i32
      cf.br ^bb1(%53, %52 : i32, !hw.array<16xi8>)
    ^bb3(%54: !hw.array<16xi8>, %55: i1):  // pred: ^bb1
      llhd.yield %54, %55 : !hw.array<16xi8>, i1
    }
    %35 = comb.mux bin %34#1, %34#0, %arr : !hw.array<16xi8>
    %arr = seq.firreg %35 clock %33 : !hw.array<16xi8>
    hw.output %sum_out : i12
  }
}
