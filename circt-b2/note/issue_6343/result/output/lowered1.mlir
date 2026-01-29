module {
  func.func private @llvm.smax.i32(i32, i32) -> i32
  func.func @mlir_func_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E7conv_2D(%arg0: memref<784xi32, 1>, %arg1: memref<147xi32, 1>, %arg2: memref<3xi32, 1>, %arg3: memref<363xi32, 1>, %arg4: memref<375xi32, 1>, %arg5: memref<5xi32, 1>, %arg6: memref<245xi32, 1>, %arg7: memref<875xi32, 1>, %arg8: memref<7xi32, 1>, %arg9: memref<63xi32, 1>, %arg10: memref<630xi32, 1>, %arg11: memref<10xi32, 1>, %arg12: memref<10xi32, 1>) {
    %c65536_i64 = arith.constant 65536 : i64
    %c2_i64 = arith.constant 2 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    scf.for %arg13 = %c0 to %c3 step %c1 {
      %c0_6 = arith.constant 0 : index
      %c11 = arith.constant 11 : index
      %c1_7 = arith.constant 1 : index
      scf.for %arg14 = %c0_6 to %c11 step %c1_7 {
        %c0_8 = arith.constant 0 : index
        %c11_9 = arith.constant 11 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg15 = %c0_8 to %c11_9 step %c1_10 {
          %c0_11 = arith.constant 0 : index
          %c7_12 = arith.constant 7 : index
          %c1_13 = arith.constant 1 : index
          %0 = scf.for %arg16 = %c0_11 to %c7_12 step %c1_13 iter_args(%arg17 = %c0_i32) -> (i32) {
            %c0_15 = arith.constant 0 : index
            %c7_16 = arith.constant 7 : index
            %c1_17 = arith.constant 1 : index
            %8 = scf.for %arg18 = %c0_15 to %c7_16 step %c1_17 iter_args(%arg19 = %arg17) -> (i32) {
              %c21 = arith.constant 21 : index
              %9 = arith.muli %arg16, %c21 overflow<nsw> : index
              %10 = arith.addi %9, %arg13 : index
              %c3_18 = arith.constant 3 : index
              %11 = arith.muli %arg18, %c3_18 overflow<nsw> : index
              %12 = arith.addi %10, %11 : index
              %13 = memref.load %arg1[%12] : memref<147xi32, 1>
              %14 = arith.extsi %13 : i32 to i64
              %c2 = arith.constant 2 : index
              %15 = arith.muli %arg14, %c2 overflow<nsw> : index
              %16 = arith.addi %arg16, %15 : index
              %c28 = arith.constant 28 : index
              %17 = arith.muli %16, %c28 overflow<nsw> : index
              %c2_19 = arith.constant 2 : index
              %18 = arith.muli %arg15, %c2_19 overflow<nsw> : index
              %19 = arith.addi %17, %18 : index
              %20 = arith.addi %19, %arg18 : index
              %21 = memref.load %arg0[%20] : memref<784xi32, 1>
              %22 = arith.extsi %21 : i32 to i64
              %23 = arith.muli %14, %22 : i64
              %24 = arith.divsi %23, %c32768_i64 : i64
              %25 = arith.remsi %24, %c2_i64 : i64
              %26 = arith.divsi %23, %c65536_i64 : i64
              %27 = arith.addi %25, %26 : i64
              %28 = arith.trunci %27 : i64 to i32
              %29 = arith.addi %arg19, %28 : i32
              scf.yield %29 : i32
            }
            scf.yield %8 : i32
          }
          %1 = memref.load %arg2[%arg13] : memref<3xi32, 1>
          %2 = arith.addi %1, %0 : i32
          %3 = func.call @llvm.smax.i32(%2, %c0_i32) : (i32, i32) -> i32
          %c33 = arith.constant 33 : index
          %4 = arith.muli %arg14, %c33 overflow<nsw> : index
          %5 = arith.addi %4, %arg13 : index
          %c3_14 = arith.constant 3 : index
          %6 = arith.muli %arg15, %c3_14 overflow<nsw> : index
          %7 = arith.addi %5, %6 : index
          memref.store %3, %arg3[%7] : memref<363xi32, 1>
        }
      }
    }
    %c0_0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %c1_1 = arith.constant 1 : index
    scf.for %arg13 = %c0_0 to %c5 step %c1_1 {
      %c0_6 = arith.constant 0 : index
      %c7_7 = arith.constant 7 : index
      %c1_8 = arith.constant 1 : index
      scf.for %arg14 = %c0_6 to %c7_7 step %c1_8 {
        %c0_9 = arith.constant 0 : index
        %c7_10 = arith.constant 7 : index
        %c1_11 = arith.constant 1 : index
        scf.for %arg15 = %c0_9 to %c7_10 step %c1_11 {
          %c0_12 = arith.constant 0 : index
          %c5_13 = arith.constant 5 : index
          %c1_14 = arith.constant 1 : index
          %0 = scf.for %arg16 = %c0_12 to %c5_13 step %c1_14 iter_args(%arg17 = %c0_i32) -> (i32) {
            %c0_16 = arith.constant 0 : index
            %c5_17 = arith.constant 5 : index
            %c1_18 = arith.constant 1 : index
            %8 = scf.for %arg18 = %c0_16 to %c5_17 step %c1_18 iter_args(%arg19 = %arg17) -> (i32) {
              %c0_19 = arith.constant 0 : index
              %c3_20 = arith.constant 3 : index
              %c1_21 = arith.constant 1 : index
              %9 = scf.for %arg20 = %c0_19 to %c3_20 step %c1_21 iter_args(%arg21 = %arg19) -> (i32) {
                %c75 = arith.constant 75 : index
                %10 = arith.muli %arg16, %c75 overflow<nsw> : index
                %11 = arith.addi %10, %arg13 : index
                %c15 = arith.constant 15 : index
                %12 = arith.muli %arg18, %c15 overflow<nsw> : index
                %13 = arith.addi %11, %12 : index
                %c5_22 = arith.constant 5 : index
                %14 = arith.muli %arg20, %c5_22 overflow<nsw> : index
                %15 = arith.addi %13, %14 : index
                %16 = memref.load %arg4[%15] : memref<375xi32, 1>
                %17 = arith.extsi %16 : i32 to i64
                %18 = arith.addi %arg18, %arg15 : index
                %c3_23 = arith.constant 3 : index
                %19 = arith.muli %18, %c3_23 overflow<nsw> : index
                %20 = arith.addi %arg16, %arg14 : index
                %c33 = arith.constant 33 : index
                %21 = arith.muli %20, %c33 overflow<nsw> : index
                %22 = arith.addi %19, %21 : index
                %23 = arith.addi %22, %arg20 : index
                %24 = memref.load %arg3[%23] : memref<363xi32, 1>
                %25 = arith.extsi %24 : i32 to i64
                %26 = arith.muli %17, %25 : i64
                %27 = arith.divsi %26, %c32768_i64 : i64
                %28 = arith.remsi %27, %c2_i64 : i64
                %29 = arith.divsi %26, %c65536_i64 : i64
                %30 = arith.addi %28, %29 : i64
                %31 = arith.trunci %30 : i64 to i32
                %32 = arith.addi %arg21, %31 : i32
                scf.yield %32 : i32
              }
              scf.yield %9 : i32
            }
            scf.yield %8 : i32
          }
          %1 = memref.load %arg5[%arg13] : memref<5xi32, 1>
          %2 = arith.addi %1, %0 : i32
          %3 = func.call @llvm.smax.i32(%2, %c0_i32) : (i32, i32) -> i32
          %c35 = arith.constant 35 : index
          %4 = arith.muli %arg14, %c35 overflow<nsw> : index
          %5 = arith.addi %4, %arg13 : index
          %c5_15 = arith.constant 5 : index
          %6 = arith.muli %arg15, %c5_15 overflow<nsw> : index
          %7 = arith.addi %5, %6 : index
          memref.store %3, %arg6[%7] : memref<245xi32, 1>
        }
      }
    }
    %c0_2 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c1_3 = arith.constant 1 : index
    scf.for %arg13 = %c0_2 to %c7 step %c1_3 {
      %c0_6 = arith.constant 0 : index
      %c3_7 = arith.constant 3 : index
      %c1_8 = arith.constant 1 : index
      scf.for %arg14 = %c0_6 to %c3_7 step %c1_8 {
        %c0_9 = arith.constant 0 : index
        %c3_10 = arith.constant 3 : index
        %c1_11 = arith.constant 1 : index
        scf.for %arg15 = %c0_9 to %c3_10 step %c1_11 {
          %c0_12 = arith.constant 0 : index
          %c5_13 = arith.constant 5 : index
          %c1_14 = arith.constant 1 : index
          %0 = scf.for %arg16 = %c0_12 to %c5_13 step %c1_14 iter_args(%arg17 = %c0_i32) -> (i32) {
            %c0_16 = arith.constant 0 : index
            %c5_17 = arith.constant 5 : index
            %c1_18 = arith.constant 1 : index
            %8 = scf.for %arg18 = %c0_16 to %c5_17 step %c1_18 iter_args(%arg19 = %arg17) -> (i32) {
              %c0_19 = arith.constant 0 : index
              %c5_20 = arith.constant 5 : index
              %c1_21 = arith.constant 1 : index
              %9 = scf.for %arg20 = %c0_19 to %c5_20 step %c1_21 iter_args(%arg21 = %arg19) -> (i32) {
                %c175 = arith.constant 175 : index
                %10 = arith.muli %arg16, %c175 overflow<nsw> : index
                %11 = arith.addi %10, %arg13 : index
                %c35 = arith.constant 35 : index
                %12 = arith.muli %arg18, %c35 overflow<nsw> : index
                %13 = arith.addi %11, %12 : index
                %c7_22 = arith.constant 7 : index
                %14 = arith.muli %arg20, %c7_22 overflow<nsw> : index
                %15 = arith.addi %13, %14 : index
                %16 = memref.load %arg7[%15] : memref<875xi32, 1>
                %17 = arith.extsi %16 : i32 to i64
                %18 = arith.addi %arg18, %arg15 : index
                %c5_23 = arith.constant 5 : index
                %19 = arith.muli %18, %c5_23 overflow<nsw> : index
                %20 = arith.addi %arg16, %arg14 : index
                %c35_24 = arith.constant 35 : index
                %21 = arith.muli %20, %c35_24 overflow<nsw> : index
                %22 = arith.addi %19, %21 : index
                %23 = arith.addi %22, %arg20 : index
                %24 = memref.load %arg6[%23] : memref<245xi32, 1>
                %25 = arith.extsi %24 : i32 to i64
                %26 = arith.muli %17, %25 : i64
                %27 = arith.divsi %26, %c32768_i64 : i64
                %28 = arith.remsi %27, %c2_i64 : i64
                %29 = arith.divsi %26, %c65536_i64 : i64
                %30 = arith.addi %28, %29 : i64
                %31 = arith.trunci %30 : i64 to i32
                %32 = arith.addi %arg21, %31 : i32
                scf.yield %32 : i32
              }
              scf.yield %9 : i32
            }
            scf.yield %8 : i32
          }
          %1 = memref.load %arg8[%arg13] : memref<7xi32, 1>
          %2 = arith.addi %1, %0 : i32
          %3 = func.call @llvm.smax.i32(%2, %c0_i32) : (i32, i32) -> i32
          %c21 = arith.constant 21 : index
          %4 = arith.muli %arg14, %c21 overflow<nsw> : index
          %5 = arith.addi %4, %arg13 : index
          %c7_15 = arith.constant 7 : index
          %6 = arith.muli %arg15, %c7_15 overflow<nsw> : index
          %7 = arith.addi %5, %6 : index
          memref.store %3, %arg9[%7] : memref<63xi32, 1>
        }
      }
    }
    %c0_4 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1_5 = arith.constant 1 : index
    scf.for %arg13 = %c0_4 to %c10 step %c1_5 {
      %c0_6 = arith.constant 0 : index
      %c3_7 = arith.constant 3 : index
      %c1_8 = arith.constant 1 : index
      %0 = scf.for %arg14 = %c0_6 to %c3_7 step %c1_8 iter_args(%arg15 = %c0_i32) -> (i32) {
        %c0_9 = arith.constant 0 : index
        %c3_10 = arith.constant 3 : index
        %c1_11 = arith.constant 1 : index
        %4 = scf.for %arg16 = %c0_9 to %c3_10 step %c1_11 iter_args(%arg17 = %arg15) -> (i32) {
          %c0_12 = arith.constant 0 : index
          %c7_13 = arith.constant 7 : index
          %c1_14 = arith.constant 1 : index
          %5 = scf.for %arg18 = %c0_12 to %c7_13 step %c1_14 iter_args(%arg19 = %arg17) -> (i32) {
            %c210 = arith.constant 210 : index
            %6 = arith.muli %arg14, %c210 overflow<nsw> : index
            %7 = arith.addi %6, %arg13 : index
            %c70 = arith.constant 70 : index
            %8 = arith.muli %arg16, %c70 overflow<nsw> : index
            %9 = arith.addi %7, %8 : index
            %c10_15 = arith.constant 10 : index
            %10 = arith.muli %arg18, %c10_15 overflow<nsw> : index
            %11 = arith.addi %9, %10 : index
            %12 = memref.load %arg10[%11] : memref<630xi32, 1>
            %13 = arith.extsi %12 : i32 to i64
            %c7_16 = arith.constant 7 : index
            %14 = arith.muli %arg16, %c7_16 overflow<nsw> : index
            %c21 = arith.constant 21 : index
            %15 = arith.muli %arg14, %c21 overflow<nsw> : index
            %16 = arith.addi %14, %15 : index
            %17 = arith.addi %16, %arg18 : index
            %18 = memref.load %arg9[%17] : memref<63xi32, 1>
            %19 = arith.extsi %18 : i32 to i64
            %20 = arith.muli %13, %19 : i64
            %21 = arith.divsi %20, %c32768_i64 : i64
            %22 = arith.remsi %21, %c2_i64 : i64
            %23 = arith.divsi %20, %c65536_i64 : i64
            %24 = arith.addi %22, %23 : i64
            %25 = arith.trunci %24 : i64 to i32
            %26 = arith.addi %arg19, %25 : i32
            scf.yield %26 : i32
          }
          scf.yield %5 : i32
        }
        scf.yield %4 : i32
      }
      %1 = memref.load %arg11[%arg13] : memref<10xi32, 1>
      %2 = arith.addi %1, %0 : i32
      %3 = func.call @llvm.smax.i32(%2, %c0_i32) : (i32, i32) -> i32
      memref.store %3, %arg12[%arg13] : memref<10xi32, 1>
    }
    return
  }
}

