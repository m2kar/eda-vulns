// Minimized test case for CIRCT bug: string type output port crash
// Original: circt-bassertion_b3788960852a_20260126_032644
// Root cause: sim::DynamicStringType not valid for hw::PortInfo

module test(output string msg);
  assign msg = "Test";
endmodule
