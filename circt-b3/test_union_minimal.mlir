// RUN: circt-opt %s --convert-moore-to-core

moore.module @TestUnionMinimal(in %arg0 : !moore.i32, out o : !moore.i32) {
  moore.output %arg0 : !moore.i32
}
