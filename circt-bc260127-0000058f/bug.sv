// Minimal reproducer: packed union type as module port crashes MooreToCore pass
// Bug: Missing type conversion for PackedUnionType in MooreToCore

typedef union packed {
  logic a;
} u;

module m(input u x);
endmodule
