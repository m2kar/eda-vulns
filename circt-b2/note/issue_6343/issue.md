# Issue #6343: MLIR lowering issue

**Author**: zzy
**Status**: OPEN
**Created**: 2023-10-25T18:50:36Z

---

HI,

I have a MLIR implementing fixed-point 4-layer convolution below

```mlir
func.func private @llvm.smax.i32(i32, i32) -> i32
func.func @mlir_func_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E7conv_2D(%arg0: memref<784xi32, 1>, %arg1: memref<147xi32, 1>, %arg2: memref<3xi32, 1>, %arg3: memref<363xi32, 1>, %arg4: memref<375xi32, 1>, %arg5: memref<5xi32, 1>, %arg6: memref<245xi32, 1>, %arg7: memref<875xi32, 1>, %arg8: memref<7xi32, 1>, %arg9: memref<63xi32, 1>, %arg10: memref<630xi32, 1>, %arg11: memref<10xi32, 1>, %arg12: memref<10xi32, 1>) {
  %c65536_i64 = arith.constant 65536 : i64
  %c2_i64 = arith.constant 2 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c0_i32 = arith.constant 0 : i32
  affine.for %arg13 = 0 to 3 {
    affine.for %arg14 = 0 to 11 {
      affine.for %arg15 = 0 to 11 {
        %0 = affine.for %arg16 = 0 to 7 iter_args(%arg17 = %c0_i32) -> (i32) {
          %4 = affine.for %arg18 = 0 to 7 iter_args(%arg19 = %arg17) -> (i32) {
            %5 = affine.load %arg1[%arg16 * 21 + %arg13 + %arg18 * 3] : memref<147xi32, 1>
            %6 = arith.extsi %5 : i32 to i64
            %7 = affine.load %arg0[(%arg16 + %arg14 * 2) * 28 + %arg15 * 2 + %arg18] : memref<784xi32, 1>
            %8 = arith.extsi %7 : i32 to i64
            %9 = arith.muli %6, %8 : i64
            %10 = arith.divsi %9, %c32768_i64 : i64
            %11 = arith.remsi %10, %c2_i64 : i64
            %12 = arith.divsi %9, %c65536_i64 : i64
            %13 = arith.addi %11, %12 : i64
            %14 = arith.trunci %13 : i64 to i32
            %15 = arith.addi %arg19, %14 : i32
            affine.yield %15 : i32
          }
          affine.yield %4 : i32
        }
        %1 = affine.load %arg2[%arg13] : memref<3xi32, 1>
        %2 = arith.addi %1, %0 : i32
        %3 = func.call @llvm.smax.i32(%2, %c0_i32) : (i32, i32) -> i32
        affine.store %3, %arg3[%arg14 * 33 + %arg13 + %arg15 * 3] : memref<363xi32, 1>
      }
    }
  }
  affine.for %arg13 = 0 to 5 {
    affine.for %arg14 = 0 to 7 {
      affine.for %arg15 = 0 to 7 {
        %0 = affine.for %arg16 = 0 to 5 iter_args(%arg17 = %c0_i32) -> (i32) {
          %4 = affine.for %arg18 = 0 to 5 iter_args(%arg19 = %arg17) -> (i32) {
            %5 = affine.for %arg20 = 0 to 3 iter_args(%arg21 = %arg19) -> (i32) {
              %6 = affine.load %arg4[%arg16 * 75 + %arg13 + %arg18 * 15 + %arg20 * 5] : memref<375xi32, 1>
              %7 = arith.extsi %6 : i32 to i64
              %8 = affine.load %arg3[(%arg18 + %arg15) * 3 + (%arg16 + %arg14) * 33 + %arg20] : memref<363xi32, 1>
              %9 = arith.extsi %8 : i32 to i64
              %10 = arith.muli %7, %9 : i64
              %11 = arith.divsi %10, %c32768_i64 : i64
              %12 = arith.remsi %11, %c2_i64 : i64
              %13 = arith.divsi %10, %c65536_i64 : i64
              %14 = arith.addi %12, %13 : i64
              %15 = arith.trunci %14 : i64 to i32
              %16 = arith.addi %arg21, %15 : i32
              affine.yield %16 : i32
            }
            affine.yield %5 : i32
          }
          affine.yield %4 : i32
        }
        %1 = affine.load %arg5[%arg13] : memref<5xi32, 1>
        %2 = arith.addi %1, %0 : i32
        %3 = func.call @llvm.smax.i32(%2, %c0_i32) : (i32, i32) -> i32
        affine.store %3, %arg6[%arg14 * 35 + %arg13 + %arg15 * 5] : memref<245xi32, 1>
      }
    }
  }
  affine.for %arg13 = 0 to 7 {
    affine.for %arg14 = 0 to 3 {
      affine.for %arg15 = 0 to 3 {
        %0 = affine.for %arg16 = 0 to 5 iter_args(%arg17 = %c0_i32) -> (i32) {
          %4 = affine.for %arg18 = 0 to 5 iter_args(%arg19 = %arg17) -> (i32) {
            %5 = affine.for %arg20 = 0 to 5 iter_args(%arg21 = %arg19) -> (i32) {
              %6 = affine.load %arg7[%arg16 * 175 + %arg13 + %arg18 * 35 + %arg20 * 7] : memref<875xi32, 1>
              %7 = arith.extsi %6 : i32 to i64
              %8 = affine.load %arg6[(%arg18 + %arg15) * 5 + (%arg16 + %arg14) * 35 + %arg20] : memref<245xi32, 1>
              %9 = arith.extsi %8 : i32 to i64
              %10 = arith.muli %7, %9 : i64
              %11 = arith.divsi %10, %c32768_i64 : i64
              %12 = arith.remsi %11, %c2_i64 : i64
              %13 = arith.divsi %10, %c65536_i64 : i64
              %14 = arith.addi %12, %13 : i64
              %15 = arith.trunci %14 : i64 to i32
              %16 = arith.addi %arg21, %15 : i32
              affine.yield %16 : i32
            }
            affine.yield %5 : i32
          }
          affine.yield %4 : i32
        }
        %1 = affine.load %arg8[%arg13] : memref<7xi32, 1>
        %2 = arith.addi %1, %0 : i32
        %3 = func.call @llvm.smax.i32(%2, %c0_i32) : (i32, i32) -> i32
        affine.store %3, %arg9[%arg14 * 21 + %arg13 + %arg15 * 7] : memref<63xi32, 1>
      }
    }
  }
  affine.for %arg13 = 0 to 10 {
    %0 = affine.for %arg14 = 0 to 3 iter_args(%arg15 = %c0_i32) -> (i32) {
      %4 = affine.for %arg16 = 0 to 3 iter_args(%arg17 = %arg15) -> (i32) {
        %5 = affine.for %arg18 = 0 to 7 iter_args(%arg19 = %arg17) -> (i32) {
          %6 = affine.load %arg10[%arg14 * 210 + %arg13 + %arg16 * 70 + %arg18 * 10] : memref<630xi32, 1>
          %7 = arith.extsi %6 : i32 to i64
          %8 = affine.load %arg9[%arg16 * 7 + %arg14 * 21 + %arg18] : memref<63xi32, 1>
          %9 = arith.extsi %8 : i32 to i64
          %10 = arith.muli %7, %9 : i64
          %11 = arith.divsi %10, %c32768_i64 : i64
          %12 = arith.remsi %11, %c2_i64 : i64
          %13 = arith.divsi %10, %c65536_i64 : i64
          %14 = arith.addi %12, %13 : i64
          %15 = arith.trunci %14 : i64 to i32
          %16 = arith.addi %arg19, %15 : i32
          affine.yield %16 : i32
        }
        affine.yield %5 : i32
      }
      affine.yield %4 : i32
    }
    %1 = affine.load %arg11[%arg13] : memref<10xi32, 1>
    %2 = arith.addi %1, %0 : i32
    %3 = func.call @llvm.smax.i32(%2, %c0_i32) : (i32, i32) -> i32
    affine.store %3, %arg12[%arg13] : memref<10xi32, 1>
    }
    return
  }
```

When I lowered it using

```bash
mlir-opt --lower-affine
mlir-opt --scf-for-to-while
circt-opt --lower-scf-to-calyx --canonicalize --verify-diagnostics
```

command lines. The back-end crashed with following error:

```
circt/llvm/llvm/include/llvm/ADT/ilist_iterator.h:138: llvm::ilist_iterator<OptionsT, IsReverse, IsConst>::reference llvm::ilist_iterator<OptionsT, IsReverse, IsConst>::operator*() const [with OptionsT = llvm::ilist_detail::node_options<mlir::Block, true, false, void>; bool IsReverse = false; bool IsConst = false; llvm::ilist_iterator<OptionsT, IsReverse, IsConst>::reference = mlir::Block&]: Assertion `!NodePtr->isKnownSentinel()' failed.
```

My guess is CIRCT can't deal with `llvm.smax`. But I am not quite sure. `llvm.smax` essentially does RELU. Normally `arith.cmpf` can do the RELU for floating-point, but Calyx cannot deal with floating-point as far as I know. So I need to use fixed-point, but `@llvm.smax` was generated unexpectedly.

Looking forwards to any suggestions.
Many thanks.

---

## Comments

**@mortbopet** (CONTRIBUTOR) - 2023-10-30T07:23:47Z

> My guess is CIRCT can't deal with llvm.smax

It's true that the function call to `llvm.smax` at some point would have to be dealth with - i imagine Calyx to be neat for doing that, actually - it should be fairly simple to integrate "function calls" in the Calyx model (implying either an instantiation of an external or library module).

Based on the error, I think it would be more likely that there is an actual bug somewhere in `scf-to-calyx`.
Regardless, this is a very big and complex example which you're probably not going to get anyone to dig into if this is the repro. So i'd suggest that you:
1. Figure out which pass is crashing, and then only report that - you're running 3 passes at the moment. This is both to handle 1) the case where it's not a CIRCT pass but instead an upstream pass, wherein the issue should be filed upstream and 2) so we know who to ping for this issue (different people are responsible for different passes).
2. Only provide the IR that feeds into the pass as found above
3. Minimize this IR to the smallest amount of lines while still triggering the issue. E.g. in your sample IR you've provided 4 multi-nested loops - I wouldn't be surprised if you'd be able to reduce this repro to only a single outer loop, and possibly also stripping away some of the inner loops and some of the many memory arguments.
