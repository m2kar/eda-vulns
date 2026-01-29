// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vbug.h for the primary calling header

#ifndef VERILATED_VBUG___024ROOT_H_
#define VERILATED_VBUG___024ROOT_H_  // guard

#include "verilated.h"


class Vbug__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vbug___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VL_IN8(clk,0,0);
    VL_IN8(a,0,0);
    VL_OUT8(b,0,0);
    VL_INOUT8(c,0,0);
    CData/*3:0*/ MixedPorts__DOT__temp_reg;
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __VicoFirstIteration;
    CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
    CData/*0:0*/ __VactContinue;
    IData/*31:0*/ __VactIterCount;
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<1> __VicoTriggered;
    VlTriggerVec<1> __VactTriggered;
    VlTriggerVec<1> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vbug__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vbug___024root(Vbug__Syms* symsp, const char* v__name);
    ~Vbug___024root();
    VL_UNCOPYABLE(Vbug___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
