module {
  hw.module @array_processor(in %clk : i1, in %start : i1, out result : i8, out done : i1) {
    %c-1_i2 = hw.constant -1 : i2
    %c-2_i2 = hw.constant -2 : i2
    %c1_i2 = hw.constant 1 : i2
    %c0_i2 = hw.constant 0 : i2
    %0 = hw.aggregate_constant [true, 96 : i8] : !hw.struct<valid: i1, data: i8>
    %1 = hw.aggregate_constant [true, 64 : i8] : !hw.struct<valid: i1, data: i8>
    %2 = hw.aggregate_constant [true, 32 : i8] : !hw.struct<valid: i1, data: i8>
    %3 = hw.aggregate_constant [true, 0 : i8] : !hw.struct<valid: i1, data: i8>
    %true = hw.constant true
    %c1_i3 = hw.constant 1 : i3
    %c-4_i3 = hw.constant -4 : i3
    %c0_i3 = hw.constant 0 : i3
    %c0_i8 = hw.constant 0 : i8
    %4 = hw.array_create %0, %1, %2, %3 : !hw.struct<valid: i1, data: i8>
    %5 = comb.xor %start, %true : i1
    %6 = seq.to_clock %clk
    %7 = comb.mux bin %start, %4, %packet_array : !hw.array<4xstruct<valid: i1, data: i8>>
    %packet_array = seq.firreg %7 clock %6 : !hw.array<4xstruct<valid: i1, data: i8>>
    %8 = hw.array_get %packet_array[%c0_i2] : !hw.array<4xstruct<valid: i1, data: i8>>, i2
    %valid = hw.struct_extract %8["valid"] : !hw.struct<valid: i1, data: i8>
    %data = hw.struct_extract %8["data"] : !hw.struct<valid: i1, data: i8>
    %9 = comb.xor %valid, %true : i1
    %10 = comb.mux %9, %c0_i8, %data : i8
    %11 = hw.array_get %packet_array[%c1_i2] : !hw.array<4xstruct<valid: i1, data: i8>>, i2
    %valid_0 = hw.struct_extract %11["valid"] : !hw.struct<valid: i1, data: i8>
    %data_1 = hw.struct_extract %11["data"] : !hw.struct<valid: i1, data: i8>
    %12 = comb.add %10, %data_1 : i8
    %13 = comb.xor %valid_0, %true : i1
    %14 = comb.mux %13, %10, %12 : i8
    %15 = hw.array_get %packet_array[%c-2_i2] : !hw.array<4xstruct<valid: i1, data: i8>>, i2
    %valid_2 = hw.struct_extract %15["valid"] : !hw.struct<valid: i1, data: i8>
    %data_3 = hw.struct_extract %15["data"] : !hw.struct<valid: i1, data: i8>
    %16 = comb.add %14, %data_3 : i8
    %17 = comb.xor %valid_2, %true : i1
    %18 = comb.mux %17, %14, %16 : i8
    %19 = hw.array_get %packet_array[%c-1_i2] : !hw.array<4xstruct<valid: i1, data: i8>>, i2
    %valid_4 = hw.struct_extract %19["valid"] : !hw.struct<valid: i1, data: i8>
    %data_5 = hw.struct_extract %19["data"] : !hw.struct<valid: i1, data: i8>
    %20 = comb.add %18, %data_5 : i8
    %21 = comb.xor %valid_4, %true : i1
    %22 = comb.mux %21, %18, %20 : i8
    %23 = comb.extract %counter from 2 : (i3) -> i1
    %24 = comb.add %counter, %c1_i3 : i3
    %25 = comb.or %23, %start : i1
    %26 = comb.mux %25, %c0_i3, %24 : i3
    %27 = comb.and %5, %23 : i1
    %28 = comb.mux bin %27, %counter, %26 : i3
    %counter = seq.firreg %28 clock %6 : i3
    %29 = comb.icmp eq %counter, %c-4_i3 : i3
    hw.output %22, %29 : i8, i1
  }
}
