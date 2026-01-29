module {
  func.func @f() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index

    %0:2 = scf.while (%arg0 = %c0, %arg1 = %c0_i32)
        : (index, i32) -> (index, i32) {
      %1 = arith.cmpi slt, %arg0, %c10 : index
      scf.condition(%1) %arg0, %arg1 : index, i32
    } do {
    ^bb0(%arg0: index, %arg1: i32):
      %1 = arith.addi %arg0, %c1 : index
      %2 = arith.addi %arg1, %c1_i32 : i32
      scf.yield %1, %2 : index, i32
    }

    return %0#1 : i32
  }
}