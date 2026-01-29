// Minimal test case for arcilator crash with inout ports
// Bug: arcilator crashes on modules with inout (bidirectional) ports
// Root cause: LowerState pass cannot create StateType for llhd.ref type

module MinimalInout(inout logic c);
endmodule
