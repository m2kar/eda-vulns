module {
  hw.module @test_module(in %signed_port : i8, in %unsigned_port : i8, out out_port : i8, in %io_port : !llhd.ref<i1>) {
    %0 = comb.icmp ugt %unsigned_port, %signed_port : i8
    %1 = comb.mux %0, %unsigned_port, %signed_port : i8
    hw.output %1 : i8
  }
}
