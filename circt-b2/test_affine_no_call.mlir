// 反例测例 - Issue #6343
// 与 test_affine.mlir 相同的结构，但移除了 func.call @llvm.smax.i32
// 用于证明问题确实是由 func.call 引发的
//
// 预期结果: 此测例应该能正常通过 lower-scf-to-calyx
//
// 复现命令:
//   mlir-opt --lower-affine test_affine_no_call.mlir -o lowered1_no_call.mlir
//   mlir-opt --scf-for-to-while lowered1_no_call.mlir -o lowered2_no_call.mlir
//   circt-opt --pass-pipeline='builtin.module(lower-scf-to-calyx{top-level-function=test_func})' lowered2_no_call.mlir

// 注意: 没有 func.func private @llvm.smax.i32 声明

func.func @test_func(%arg0: memref<16xi32>, %arg1: memref<4xi32>) {
  %c0_i32 = arith.constant 0 : i32

  affine.for %i = 0 to 4 {
    %0 = affine.for %j = 0 to 4 iter_args(%acc = %c0_i32) -> (i32) {
      %val = affine.load %arg0[%j] : memref<16xi32>
      %new_acc = arith.addi %acc, %val : i32
      affine.yield %new_acc : i32
    }
    // 直接存储结果，不调用外部函数
    // 原代码: %1 = func.call @llvm.smax.i32(%0, %c0_i32) : (i32, i32) -> i32
    affine.store %0, %arg1[%i] : memref<4xi32>
  }
  return
}
