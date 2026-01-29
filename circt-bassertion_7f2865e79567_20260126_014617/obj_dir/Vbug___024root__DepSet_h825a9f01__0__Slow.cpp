// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vbug.h for the primary calling header

#include "Vbug__pch.h"
#include "Vbug___024root.h"

VL_ATTR_COLD void Vbug___024root___eval_static(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___eval_static\n"); );
}

VL_ATTR_COLD void Vbug___024root___eval_initial__TOP(Vbug___024root* vlSelf);

VL_ATTR_COLD void Vbug___024root___eval_initial(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___eval_initial\n"); );
    // Body
    Vbug___024root___eval_initial__TOP(vlSelf);
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = vlSelf->clk;
}

VL_ATTR_COLD void Vbug___024root___eval_initial__TOP(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___eval_initial__TOP\n"); );
    // Body
    vlSelf->b = 0U;
}

VL_ATTR_COLD void Vbug___024root___eval_final(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___eval_final\n"); );
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbug___024root___dump_triggers__stl(Vbug___024root* vlSelf);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vbug___024root___eval_phase__stl(Vbug___024root* vlSelf);

VL_ATTR_COLD void Vbug___024root___eval_settle(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___eval_settle\n"); );
    // Init
    IData/*31:0*/ __VstlIterCount;
    CData/*0:0*/ __VstlContinue;
    // Body
    __VstlIterCount = 0U;
    vlSelf->__VstlFirstIteration = 1U;
    __VstlContinue = 1U;
    while (__VstlContinue) {
        if (VL_UNLIKELY((0x64U < __VstlIterCount))) {
#ifdef VL_DEBUG
            Vbug___024root___dump_triggers__stl(vlSelf);
#endif
            VL_FATAL_MT("/home/zhiqing/edazz/eda-vulns/circt-bassertion_7f2865e79567_20260126_014617/origin/bug.sv", 1, "", "Settle region did not converge.");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        __VstlContinue = 0U;
        if (Vbug___024root___eval_phase__stl(vlSelf)) {
            __VstlContinue = 1U;
        }
        vlSelf->__VstlFirstIteration = 0U;
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbug___024root___dump_triggers__stl(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VstlTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        VL_DBG_MSGF("         'stl' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

void Vbug___024root___ico_sequent__TOP__0(Vbug___024root* vlSelf);

VL_ATTR_COLD void Vbug___024root___eval_stl(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___eval_stl\n"); );
    // Body
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        Vbug___024root___ico_sequent__TOP__0(vlSelf);
    }
}

VL_ATTR_COLD void Vbug___024root___eval_triggers__stl(Vbug___024root* vlSelf);

VL_ATTR_COLD bool Vbug___024root___eval_phase__stl(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___eval_phase__stl\n"); );
    // Init
    CData/*0:0*/ __VstlExecute;
    // Body
    Vbug___024root___eval_triggers__stl(vlSelf);
    __VstlExecute = vlSelf->__VstlTriggered.any();
    if (__VstlExecute) {
        Vbug___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbug___024root___dump_triggers__ico(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___dump_triggers__ico\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VicoTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VicoTriggered.word(0U))) {
        VL_DBG_MSGF("         'ico' region trigger index 0 is active: Internal 'ico' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbug___024root___dump_triggers__act(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VactTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge clk)\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbug___024root___dump_triggers__nba(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___dump_triggers__nba\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VnbaTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge clk)\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vbug___024root___ctor_var_reset(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___ctor_var_reset\n"); );
    // Body
    vlSelf->a = VL_RAND_RESET_I(1);
    vlSelf->b = VL_RAND_RESET_I(1);
    vlSelf->c = VL_RAND_RESET_I(1);
    vlSelf->clk = VL_RAND_RESET_I(1);
    vlSelf->MixedPorts__DOT__temp_reg = VL_RAND_RESET_I(4);
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = VL_RAND_RESET_I(1);
}
