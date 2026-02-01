// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vbug.h for the primary calling header

#include "Vbug__pch.h"
#include "Vbug__Syms.h"
#include "Vbug___024root.h"

void Vbug___024root___ctor_var_reset(Vbug___024root* vlSelf);

Vbug___024root::Vbug___024root(Vbug__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , vlSymsp{symsp}
 {
    // Reset structure values
    Vbug___024root___ctor_var_reset(this);
}

void Vbug___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

Vbug___024root::~Vbug___024root() {
}
