
mlir-opt --lower-affine --scf-for-to-while test_affine.mlir | circt-opt --pass-pipeline='builtin.module(lower-scf-to-calyx{top-level-function=test_func})'