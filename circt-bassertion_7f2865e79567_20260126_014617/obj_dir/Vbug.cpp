// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vbug__pch.h"

//============================================================
// Constructors

Vbug::Vbug(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vbug__Syms(contextp(), _vcname__, this)}
    , clk{vlSymsp->TOP.clk}
    , a{vlSymsp->TOP.a}
    , b{vlSymsp->TOP.b}
    , c{vlSymsp->TOP.c}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

Vbug::Vbug(const char* _vcname__)
    : Vbug(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vbug::~Vbug() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vbug___024root___eval_debug_assertions(Vbug___024root* vlSelf);
#endif  // VL_DEBUG
void Vbug___024root___eval_static(Vbug___024root* vlSelf);
void Vbug___024root___eval_initial(Vbug___024root* vlSelf);
void Vbug___024root___eval_settle(Vbug___024root* vlSelf);
void Vbug___024root___eval(Vbug___024root* vlSelf);

void Vbug::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vbug::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vbug___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        vlSymsp->__Vm_didInit = true;
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vbug___024root___eval_static(&(vlSymsp->TOP));
        Vbug___024root___eval_initial(&(vlSymsp->TOP));
        Vbug___024root___eval_settle(&(vlSymsp->TOP));
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vbug___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vbug::eventsPending() { return false; }

uint64_t Vbug::nextTimeSlot() {
    VL_FATAL_MT(__FILE__, __LINE__, "", "%Error: No delays in the design");
    return 0;
}

//============================================================
// Utilities

const char* Vbug::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vbug___024root___eval_final(Vbug___024root* vlSelf);

VL_ATTR_COLD void Vbug::final() {
    Vbug___024root___eval_final(&(vlSymsp->TOP));
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vbug::hierName() const { return vlSymsp->name(); }
const char* Vbug::modelName() const { return "Vbug"; }
unsigned Vbug::threads() const { return 1; }
void Vbug::prepareClone() const { contextp()->prepareClone(); }
void Vbug::atClone() const {
    contextp()->threadPoolpOnClone();
}

//============================================================
// Trace configuration

VL_ATTR_COLD void Vbug::trace(VerilatedVcdC* tfp, int levels, int options) {
    vl_fatal(__FILE__, __LINE__, __FILE__,"'Vbug::trace()' called on model that was Verilated without --trace option");
}
