module {
  hw.module @test(in %clk : i1) {
    %c-1_i2 = hw.constant -1 : i2
    %c0_i9 = hw.constant 0 : i9
    %c0_i32 = hw.constant 0 : i32
    %0 = hw.bitcast %c0_i32 : (i32) -> !hw.array<4xi8>
    %1 = hw.bitcast %0 : (!hw.array<4xi8>) -> i32
    %2 = hw.bitcast %1 : (i32) -> !hw.array<4xi8>
    %3 = hw.bitcast %c0_i9 : (i9) -> !hw.struct<valid: i1, data: i8>
    %4 = hw.bitcast %7 : (!hw.struct<valid: i1, data: i8>) -> i9
    %5 = hw.bitcast %4 : (i9) -> !hw.struct<valid: i1, data: i8>
    %6 = hw.array_get %2[%c-1_i2] : !hw.array<4xi8>, i2
    %7 = hw.struct_inject %5["data"], %6 : !hw.struct<valid: i1, data: i8>
    %valid = hw.struct_extract %5["valid"] : !hw.struct<valid: i1, data: i8>
    %8 = seq.to_clock %clk
    %q = seq.firreg %valid clock %8 : i1
    hw.output
  }
}

