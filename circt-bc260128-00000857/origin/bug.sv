// Minimal reproducer: sim.fmt.literal legalization failure in arcilator
// Trigger: Immediate assertion with $error() generates orphaned sim.fmt.literal
module M(input q);
  always @(*) assert(q) else $error("");
endmodule
