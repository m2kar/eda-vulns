module {
  func.func private @llvm.smax.i32(i32, i32) -> i32
  func.func @mlir_func_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E7conv_2D(%arg0: memref<784xi32, 1>, %arg1: memref<147xi32, 1>, %arg2: memref<3xi32, 1>, %arg3: memref<363xi32, 1>, %arg4: memref<375xi32, 1>, %arg5: memref<5xi32, 1>, %arg6: memref<245xi32, 1>, %arg7: memref<875xi32, 1>, %arg8: memref<7xi32, 1>, %arg9: memref<63xi32, 1>, %arg10: memref<630xi32, 1>, %arg11: memref<10xi32, 1>, %arg12: memref<10xi32, 1>) {
    %c70 = arith.constant 70 : index
    %c210 = arith.constant 210 : index
    %c10 = arith.constant 10 : index
    %c175 = arith.constant 175 : index
    %c35 = arith.constant 35 : index
    %c15 = arith.constant 15 : index
    %c75 = arith.constant 75 : index
    %c5 = arith.constant 5 : index
    %c33 = arith.constant 33 : index
    %c28 = arith.constant 28 : index
    %c2 = arith.constant 2 : index
    %c21 = arith.constant 21 : index
    %c7 = arith.constant 7 : index
    %c11 = arith.constant 11 : index
    %c65536_i64 = arith.constant 65536 : i64
    %c2_i64 = arith.constant 2 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %0 = scf.while (%arg13 = %c0) : (index) -> index {
      %4 = arith.cmpi slt, %arg13, %c3 : index
      scf.condition(%4) %arg13 : index
    } do {
    ^bb0(%arg13: index):
      %4 = arith.addi %arg13, %c1 : index
      %5 = scf.while (%arg14 = %c0) : (index) -> index {
        %6 = arith.cmpi slt, %arg14, %c11 : index
        scf.condition(%6) %arg14 : index
      } do {
      ^bb0(%arg14: index):
        %6 = arith.addi %arg14, %c1 : index
        %7 = scf.while (%arg15 = %c0) : (index) -> index {
          %8 = arith.cmpi slt, %arg15, %c11 : index
          scf.condition(%8) %arg15 : index
        } do {
        ^bb0(%arg15: index):
          %8 = arith.addi %arg15, %c1 : index
          %9:2 = scf.while (%arg16 = %c0, %arg17 = %c0_i32) : (index, i32) -> (index, i32) {
            %17 = arith.cmpi slt, %arg16, %c7 : index
            scf.condition(%17) %arg16, %arg17 : index, i32
          } do {
          ^bb0(%arg16: index, %arg17: i32):
            %17 = arith.addi %arg16, %c1 : index
            %18:2 = scf.while (%arg18 = %c0, %arg19 = %arg17) : (index, i32) -> (index, i32) {
              %19 = arith.cmpi slt, %arg18, %c7 : index
              scf.condition(%19) %arg18, %arg19 : index, i32
            } do {
            ^bb0(%arg18: index, %arg19: i32):
              %19 = arith.addi %arg18, %c1 : index
              %20 = arith.muli %arg16, %c21 overflow<nsw> : index
              %21 = arith.addi %20, %arg13 : index
              %22 = arith.muli %arg18, %c3 overflow<nsw> : index
              %23 = arith.addi %21, %22 : index
              %24 = memref.load %arg1[%23] : memref<147xi32, 1>
              %25 = arith.extsi %24 : i32 to i64
              %26 = arith.muli %arg14, %c2 overflow<nsw> : index
              %27 = arith.addi %arg16, %26 : index
              %28 = arith.muli %27, %c28 overflow<nsw> : index
              %29 = arith.muli %arg15, %c2 overflow<nsw> : index
              %30 = arith.addi %28, %29 : index
              %31 = arith.addi %30, %arg18 : index
              %32 = memref.load %arg0[%31] : memref<784xi32, 1>
              %33 = arith.extsi %32 : i32 to i64
              %34 = arith.muli %25, %33 : i64
              %35 = arith.divsi %34, %c32768_i64 : i64
              %36 = arith.remsi %35, %c2_i64 : i64
              %37 = arith.divsi %34, %c65536_i64 : i64
              %38 = arith.addi %36, %37 : i64
              %39 = arith.trunci %38 : i64 to i32
              %40 = arith.addi %arg19, %39 : i32
              scf.yield %19, %40 : index, i32
            }
            scf.yield %17, %18#1 : index, i32
          }
          %10 = memref.load %arg2[%arg13] : memref<3xi32, 1>
          %11 = arith.addi %10, %9#1 : i32
          %12 = func.call @llvm.smax.i32(%11, %c0_i32) : (i32, i32) -> i32
          %13 = arith.muli %arg14, %c33 overflow<nsw> : index
          %14 = arith.addi %13, %arg13 : index
          %15 = arith.muli %arg15, %c3 overflow<nsw> : index
          %16 = arith.addi %14, %15 : index
          memref.store %12, %arg3[%16] : memref<363xi32, 1>
          scf.yield %8 : index
        }
        scf.yield %6 : index
      }
      scf.yield %4 : index
    }
    %1 = scf.while (%arg13 = %c0) : (index) -> index {
      %4 = arith.cmpi slt, %arg13, %c5 : index
      scf.condition(%4) %arg13 : index
    } do {
    ^bb0(%arg13: index):
      %4 = arith.addi %arg13, %c1 : index
      %5 = scf.while (%arg14 = %c0) : (index) -> index {
        %6 = arith.cmpi slt, %arg14, %c7 : index
        scf.condition(%6) %arg14 : index
      } do {
      ^bb0(%arg14: index):
        %6 = arith.addi %arg14, %c1 : index
        %7 = scf.while (%arg15 = %c0) : (index) -> index {
          %8 = arith.cmpi slt, %arg15, %c7 : index
          scf.condition(%8) %arg15 : index
        } do {
        ^bb0(%arg15: index):
          %8 = arith.addi %arg15, %c1 : index
          %9:2 = scf.while (%arg16 = %c0, %arg17 = %c0_i32) : (index, i32) -> (index, i32) {
            %17 = arith.cmpi slt, %arg16, %c5 : index
            scf.condition(%17) %arg16, %arg17 : index, i32
          } do {
          ^bb0(%arg16: index, %arg17: i32):
            %17 = arith.addi %arg16, %c1 : index
            %18:2 = scf.while (%arg18 = %c0, %arg19 = %arg17) : (index, i32) -> (index, i32) {
              %19 = arith.cmpi slt, %arg18, %c5 : index
              scf.condition(%19) %arg18, %arg19 : index, i32
            } do {
            ^bb0(%arg18: index, %arg19: i32):
              %19 = arith.addi %arg18, %c1 : index
              %20:2 = scf.while (%arg20 = %c0, %arg21 = %arg19) : (index, i32) -> (index, i32) {
                %21 = arith.cmpi slt, %arg20, %c3 : index
                scf.condition(%21) %arg20, %arg21 : index, i32
              } do {
              ^bb0(%arg20: index, %arg21: i32):
                %21 = arith.addi %arg20, %c1 : index
                %22 = arith.muli %arg16, %c75 overflow<nsw> : index
                %23 = arith.addi %22, %arg13 : index
                %24 = arith.muli %arg18, %c15 overflow<nsw> : index
                %25 = arith.addi %23, %24 : index
                %26 = arith.muli %arg20, %c5 overflow<nsw> : index
                %27 = arith.addi %25, %26 : index
                %28 = memref.load %arg4[%27] : memref<375xi32, 1>
                %29 = arith.extsi %28 : i32 to i64
                %30 = arith.addi %arg18, %arg15 : index
                %31 = arith.muli %30, %c3 overflow<nsw> : index
                %32 = arith.addi %arg16, %arg14 : index
                %33 = arith.muli %32, %c33 overflow<nsw> : index
                %34 = arith.addi %31, %33 : index
                %35 = arith.addi %34, %arg20 : index
                %36 = memref.load %arg3[%35] : memref<363xi32, 1>
                %37 = arith.extsi %36 : i32 to i64
                %38 = arith.muli %29, %37 : i64
                %39 = arith.divsi %38, %c32768_i64 : i64
                %40 = arith.remsi %39, %c2_i64 : i64
                %41 = arith.divsi %38, %c65536_i64 : i64
                %42 = arith.addi %40, %41 : i64
                %43 = arith.trunci %42 : i64 to i32
                %44 = arith.addi %arg21, %43 : i32
                scf.yield %21, %44 : index, i32
              }
              scf.yield %19, %20#1 : index, i32
            }
            scf.yield %17, %18#1 : index, i32
          }
          %10 = memref.load %arg5[%arg13] : memref<5xi32, 1>
          %11 = arith.addi %10, %9#1 : i32
          %12 = func.call @llvm.smax.i32(%11, %c0_i32) : (i32, i32) -> i32
          %13 = arith.muli %arg14, %c35 overflow<nsw> : index
          %14 = arith.addi %13, %arg13 : index
          %15 = arith.muli %arg15, %c5 overflow<nsw> : index
          %16 = arith.addi %14, %15 : index
          memref.store %12, %arg6[%16] : memref<245xi32, 1>
          scf.yield %8 : index
        }
        scf.yield %6 : index
      }
      scf.yield %4 : index
    }
    %2 = scf.while (%arg13 = %c0) : (index) -> index {
      %4 = arith.cmpi slt, %arg13, %c7 : index
      scf.condition(%4) %arg13 : index
    } do {
    ^bb0(%arg13: index):
      %4 = arith.addi %arg13, %c1 : index
      %5 = scf.while (%arg14 = %c0) : (index) -> index {
        %6 = arith.cmpi slt, %arg14, %c3 : index
        scf.condition(%6) %arg14 : index
      } do {
      ^bb0(%arg14: index):
        %6 = arith.addi %arg14, %c1 : index
        %7 = scf.while (%arg15 = %c0) : (index) -> index {
          %8 = arith.cmpi slt, %arg15, %c3 : index
          scf.condition(%8) %arg15 : index
        } do {
        ^bb0(%arg15: index):
          %8 = arith.addi %arg15, %c1 : index
          %9:2 = scf.while (%arg16 = %c0, %arg17 = %c0_i32) : (index, i32) -> (index, i32) {
            %17 = arith.cmpi slt, %arg16, %c5 : index
            scf.condition(%17) %arg16, %arg17 : index, i32
          } do {
          ^bb0(%arg16: index, %arg17: i32):
            %17 = arith.addi %arg16, %c1 : index
            %18:2 = scf.while (%arg18 = %c0, %arg19 = %arg17) : (index, i32) -> (index, i32) {
              %19 = arith.cmpi slt, %arg18, %c5 : index
              scf.condition(%19) %arg18, %arg19 : index, i32
            } do {
            ^bb0(%arg18: index, %arg19: i32):
              %19 = arith.addi %arg18, %c1 : index
              %20:2 = scf.while (%arg20 = %c0, %arg21 = %arg19) : (index, i32) -> (index, i32) {
                %21 = arith.cmpi slt, %arg20, %c5 : index
                scf.condition(%21) %arg20, %arg21 : index, i32
              } do {
              ^bb0(%arg20: index, %arg21: i32):
                %21 = arith.addi %arg20, %c1 : index
                %22 = arith.muli %arg16, %c175 overflow<nsw> : index
                %23 = arith.addi %22, %arg13 : index
                %24 = arith.muli %arg18, %c35 overflow<nsw> : index
                %25 = arith.addi %23, %24 : index
                %26 = arith.muli %arg20, %c7 overflow<nsw> : index
                %27 = arith.addi %25, %26 : index
                %28 = memref.load %arg7[%27] : memref<875xi32, 1>
                %29 = arith.extsi %28 : i32 to i64
                %30 = arith.addi %arg18, %arg15 : index
                %31 = arith.muli %30, %c5 overflow<nsw> : index
                %32 = arith.addi %arg16, %arg14 : index
                %33 = arith.muli %32, %c35 overflow<nsw> : index
                %34 = arith.addi %31, %33 : index
                %35 = arith.addi %34, %arg20 : index
                %36 = memref.load %arg6[%35] : memref<245xi32, 1>
                %37 = arith.extsi %36 : i32 to i64
                %38 = arith.muli %29, %37 : i64
                %39 = arith.divsi %38, %c32768_i64 : i64
                %40 = arith.remsi %39, %c2_i64 : i64
                %41 = arith.divsi %38, %c65536_i64 : i64
                %42 = arith.addi %40, %41 : i64
                %43 = arith.trunci %42 : i64 to i32
                %44 = arith.addi %arg21, %43 : i32
                scf.yield %21, %44 : index, i32
              }
              scf.yield %19, %20#1 : index, i32
            }
            scf.yield %17, %18#1 : index, i32
          }
          %10 = memref.load %arg8[%arg13] : memref<7xi32, 1>
          %11 = arith.addi %10, %9#1 : i32
          %12 = func.call @llvm.smax.i32(%11, %c0_i32) : (i32, i32) -> i32
          %13 = arith.muli %arg14, %c21 overflow<nsw> : index
          %14 = arith.addi %13, %arg13 : index
          %15 = arith.muli %arg15, %c7 overflow<nsw> : index
          %16 = arith.addi %14, %15 : index
          memref.store %12, %arg9[%16] : memref<63xi32, 1>
          scf.yield %8 : index
        }
        scf.yield %6 : index
      }
      scf.yield %4 : index
    }
    %3 = scf.while (%arg13 = %c0) : (index) -> index {
      %4 = arith.cmpi slt, %arg13, %c10 : index
      scf.condition(%4) %arg13 : index
    } do {
    ^bb0(%arg13: index):
      %4 = arith.addi %arg13, %c1 : index
      %5:2 = scf.while (%arg14 = %c0, %arg15 = %c0_i32) : (index, i32) -> (index, i32) {
        %9 = arith.cmpi slt, %arg14, %c3 : index
        scf.condition(%9) %arg14, %arg15 : index, i32
      } do {
      ^bb0(%arg14: index, %arg15: i32):
        %9 = arith.addi %arg14, %c1 : index
        %10:2 = scf.while (%arg16 = %c0, %arg17 = %arg15) : (index, i32) -> (index, i32) {
          %11 = arith.cmpi slt, %arg16, %c3 : index
          scf.condition(%11) %arg16, %arg17 : index, i32
        } do {
        ^bb0(%arg16: index, %arg17: i32):
          %11 = arith.addi %arg16, %c1 : index
          %12:2 = scf.while (%arg18 = %c0, %arg19 = %arg17) : (index, i32) -> (index, i32) {
            %13 = arith.cmpi slt, %arg18, %c7 : index
            scf.condition(%13) %arg18, %arg19 : index, i32
          } do {
          ^bb0(%arg18: index, %arg19: i32):
            %13 = arith.addi %arg18, %c1 : index
            %14 = arith.muli %arg14, %c210 overflow<nsw> : index
            %15 = arith.addi %14, %arg13 : index
            %16 = arith.muli %arg16, %c70 overflow<nsw> : index
            %17 = arith.addi %15, %16 : index
            %18 = arith.muli %arg18, %c10 overflow<nsw> : index
            %19 = arith.addi %17, %18 : index
            %20 = memref.load %arg10[%19] : memref<630xi32, 1>
            %21 = arith.extsi %20 : i32 to i64
            %22 = arith.muli %arg16, %c7 overflow<nsw> : index
            %23 = arith.muli %arg14, %c21 overflow<nsw> : index
            %24 = arith.addi %22, %23 : index
            %25 = arith.addi %24, %arg18 : index
            %26 = memref.load %arg9[%25] : memref<63xi32, 1>
            %27 = arith.extsi %26 : i32 to i64
            %28 = arith.muli %21, %27 : i64
            %29 = arith.divsi %28, %c32768_i64 : i64
            %30 = arith.remsi %29, %c2_i64 : i64
            %31 = arith.divsi %28, %c65536_i64 : i64
            %32 = arith.addi %30, %31 : i64
            %33 = arith.trunci %32 : i64 to i32
            %34 = arith.addi %arg19, %33 : i32
            scf.yield %13, %34 : index, i32
          }
          scf.yield %11, %12#1 : index, i32
        }
        scf.yield %9, %10#1 : index, i32
      }
      %6 = memref.load %arg11[%arg13] : memref<10xi32, 1>
      %7 = arith.addi %6, %5#1 : i32
      %8 = func.call @llvm.smax.i32(%7, %c0_i32) : (i32, i32) -> i32
      memref.store %8, %arg12[%arg13] : memref<10xi32, 1>
      scf.yield %4 : index
    }
    return
  }
}

