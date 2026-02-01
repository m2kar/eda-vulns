// Test if union type conversion works at all
// RUN: circt-opt %s --convert-moore-to-core

func.func @test() {
  %c0 = moore.constant 42 : !moore.i32
  %u = moore.union_create %c0 {fieldName = "a"} : !moore.i32 -> !moore.union<{a: i32, b: i32}>
  return
}
