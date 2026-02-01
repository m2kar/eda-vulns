// RUN: circt-opt %s --convert-moore-to-core

moore.module @TestUnion(in %arg0 : !moore.union<{a: i32, b: i32}>, out o : !moore.i32) {
  %0 = moore.union_extract %arg0, "a" : !moore.union<{a: i32, b: i32}> -> i32
  moore.output %0 : !moore.i32
}
