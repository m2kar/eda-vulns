// RUN: circt-opt %s --convert-moore-to-core

func.func @test_union(%arg0 : !moore.union<{a: i32, b: i32}>) -> !moore.i32 {
  %0 = moore.union_extract %arg0, "a" : !moore.union<{a: i32, b: i32}> -> i32
  return %0 : !moore.i32
}
