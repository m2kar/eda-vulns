// RUN: circt-opt %s --convert-moore-to-core

moore.module @TestStruct(in %arg0 : !moore.struct<{a: i32, b: i32}>, out o : !moore.i32) {
  %0 = moore.struct_extract %arg0, "a" : !moore.struct<{a: i32, b: i32}> -> !moore.i32
  moore.output %0 : !moore.i32
}
