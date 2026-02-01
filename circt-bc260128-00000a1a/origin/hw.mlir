module {
  hw.module @MixedPorts(in %a : i1, out b : i1, in %c : !llhd.ref<i1>) {
    hw.output %a : i1
  }
}
