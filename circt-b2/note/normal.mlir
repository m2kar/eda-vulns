// Minimal test case: c = 0; c = c + 1; loop 10 times

func.func @f() -> i32 {
  // 初始化c为0
  %c_init = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %c = affine.for %i = 0 to 10 iter_args(%c_iter = %c_init) -> (i32) {
    %c_new = arith.addi %c_iter, %one : i32
    affine.yield %c_new : i32
  }
  return %c : i32
}
