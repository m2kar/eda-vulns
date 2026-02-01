module {
  hw.module @test(in %clk : i1) {
    %0 = llhd.constant_time <0ns, 0d, 1e>
    %c-1_i2 = hw.constant -1 : i2
    %c0_i9 = hw.constant 0 : i9
    %c0_i32 = hw.constant 0 : i32
    %false = hw.constant false
    %true = hw.constant true
    %false_0 = hw.constant false
    %true_1 = hw.constant true
    %false_2 = hw.constant false
    %false_3 = hw.constant false
    %1 = hw.bitcast %c0_i32 : (i32) -> !hw.array<4xi8>
    %2 = hw.bitcast %1 : (!hw.array<4xi8>) -> i32
    %3 = hw.bitcast %2 : (i32) -> !hw.array<4xi8>
    %4 = hw.bitcast %c0_i9 : (i9) -> !hw.struct<valid: i1, data: i8>
    %5 = hw.bitcast %4 : (!hw.struct<valid: i1, data: i8>) -> i9
    %c-1_i9 = hw.constant -1 : i9
    %c0_i9_4 = hw.constant 0 : i9
    %c-1_i9_5 = hw.constant -1 : i9
    %c0_i9_6 = hw.constant 0 : i9
    %c0_i9_7 = hw.constant 0 : i9
    %6 = hw.bitcast %9 : (!hw.struct<valid: i1, data: i8>) -> i9
    %7 = hw.bitcast %6 : (i9) -> !hw.struct<valid: i1, data: i8>
    %8 = hw.array_get %3[%c-1_i2] : !hw.array<4xi8>, i2
    %9 = hw.struct_inject %7["data"], %8 : !hw.struct<valid: i1, data: i8>
    %valid = hw.struct_extract %7["valid"] : !hw.struct<valid: i1, data: i8>
    %10 = seq.to_clock %clk
    %q = seq.firreg %valid clock %10 : i1
    hw.output
  }
}

