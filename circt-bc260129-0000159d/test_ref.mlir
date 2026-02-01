module {
  hw.module @TestModule(in %clk : i1, in %a : i1, in %c : !llhd.ref<i1>) {
    hw.output %a : i1
  }
}
