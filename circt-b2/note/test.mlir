// 崩溃测例
// 这是简化版的 affine 格式测例，需要经过以下转换
// 运行命令: mlir-opt --lower-affine --scf-for-to-while test_affine.mlir | circt-opt --pass-pipeline='builtin.module(lower-scf-to-calyx{top-level-function=test_func})'
// 触发崩溃

func.func private @llvm.smax.i32(i32, i32) -> i32

func.func @test_func(%arg0: memref<16xi32>, %arg1: memref<4xi32>) {
  %c0_i32 = arith.constant 0 : i32

  affine.for %i = 0 to 4 {
    %0 = affine.for %j = 0 to 4 iter_args(%acc = %c0_i32) -> (i32) {
      %val = affine.load %arg0[%j] : memref<16xi32>
      %new_acc = arith.addi %acc, %val : i32
      affine.yield %new_acc : i32
    }
    // func.call 是触发 bug 的关键
    %1 = func.call @llvm.smax.i32(%0, %c0_i32) : (i32, i32) -> i32
    affine.store %1, %arg1[%i] : memref<4xi32>
  }
  return
}
