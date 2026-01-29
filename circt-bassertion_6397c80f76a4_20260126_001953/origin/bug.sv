// Minimal test case: string type output port causes assertion failure
// Bug: CIRCT crashes when a module has string type output port
// Expected: Proper error message or correct handling
module test_module(output string str_out);
endmodule
