// Minimal test case: func.call inside affine.for triggers SCFToCalyx segfault

func.func private @ext()

func.func @f() {
  affine.for %i = 0 to 1 {
    func.call @ext() : () -> ()
  }
  return
}
