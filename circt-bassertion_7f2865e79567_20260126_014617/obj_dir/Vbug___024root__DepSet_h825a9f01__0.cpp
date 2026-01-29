// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vbug.h for the primary calling header

#include "Vbug__pch.h"
#include "Vbug___024root.h"

VL_INLINE_OPT void Vbug___024root___ico_sequent__TOP__0(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___ico_sequent__TOP__0\n"); );
    // Body
    vlSelf->c = ((IData)(vlSelf->a) & (IData)(vlSelf->MixedPorts__DOT__temp_reg));
}

void Vbug___024root___eval_ico(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___eval_ico\n"); );
    // Body
    if ((1ULL & vlSelf->__VicoTriggered.word(0U))) {
        Vbug___024root___ico_sequent__TOP__0(vlSelf);
    }
}

void Vbug___024root___eval_triggers__ico(Vbug___024root* vlSelf);

bool Vbug___024root___eval_phase__ico(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___eval_phase__ico\n"); );
    // Init
    CData/*0:0*/ __VicoExecute;
    // Body
    Vbug___024root___eval_triggers__ico(vlSelf);
    __VicoExecute = vlSelf->__VicoTriggered.any();
    if (__VicoExecute) {
        Vbug___024root___eval_ico(vlSelf);
    }
    return (__VicoExecute);
}

void Vbug___024root___eval_act(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___eval_act\n"); );
}

VL_INLINE_OPT void Vbug___024root___nba_sequent__TOP__0(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___nba_sequent__TOP__0\n"); );
    // Body
    vlSelf->MixedPorts__DOT__temp_reg = vlSelf->c;
    vlSelf->c = ((IData)(vlSelf->a) & (IData)(vlSelf->MixedPorts__DOT__temp_reg));
}

void Vbug___024root___eval_nba(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___eval_nba\n"); );
    // Body
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vbug___024root___nba_sequent__TOP__0(vlSelf);
    }
}

void Vbug___024root___eval_triggers__act(Vbug___024root* vlSelf);

bool Vbug___024root___eval_phase__act(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___eval_phase__act\n"); );
    // Init
    VlTriggerVec<1> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vbug___024root___eval_triggers__act(vlSelf);
    __VactExecute = vlSelf->__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelf->__VactTriggered, vlSelf->__VnbaTriggered);
        vlSelf->__VnbaTriggered.thisOr(vlSelf->__VactTriggered);
        Vbug___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vbug___024root___eval_phase__nba(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___eval_phase__nba\n"); );
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelf->__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vbug___024root___eval_nba(vlSelf);
        vlSelf->__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbug___024root___dump_triggers__ico(Vbug___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vbug___024root___dump_triggers__nba(Vbug___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vbug___024root___dump_triggers__act(Vbug___024root* vlSelf);
#endif  // VL_DEBUG

void Vbug___024root___eval(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___eval\n"); );
    // Init
    IData/*31:0*/ __VicoIterCount;
    CData/*0:0*/ __VicoContinue;
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    __VicoIterCount = 0U;
    vlSelf->__VicoFirstIteration = 1U;
    __VicoContinue = 1U;
    while (__VicoContinue) {
        if (VL_UNLIKELY((0x64U < __VicoIterCount))) {
#ifdef VL_DEBUG
            Vbug___024root___dump_triggers__ico(vlSelf);
#endif
            VL_FATAL_MT("/home/zhiqing/edazz/eda-vulns/circt-bassertion_7f2865e79567_20260126_014617/origin/bug.sv", 1, "", "Input combinational region did not converge.");
        }
        __VicoIterCount = ((IData)(1U) + __VicoIterCount);
        __VicoContinue = 0U;
        if (Vbug___024root___eval_phase__ico(vlSelf)) {
            __VicoContinue = 1U;
        }
        vlSelf->__VicoFirstIteration = 0U;
    }
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY((0x64U < __VnbaIterCount))) {
#ifdef VL_DEBUG
            Vbug___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("/home/zhiqing/edazz/eda-vulns/circt-bassertion_7f2865e79567_20260126_014617/origin/bug.sv", 1, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelf->__VactIterCount = 0U;
        vlSelf->__VactContinue = 1U;
        while (vlSelf->__VactContinue) {
            if (VL_UNLIKELY((0x64U < vlSelf->__VactIterCount))) {
#ifdef VL_DEBUG
                Vbug___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("/home/zhiqing/edazz/eda-vulns/circt-bassertion_7f2865e79567_20260126_014617/origin/bug.sv", 1, "", "Active region did not converge.");
            }
            vlSelf->__VactIterCount = ((IData)(1U) 
                                       + vlSelf->__VactIterCount);
            vlSelf->__VactContinue = 0U;
            if (Vbug___024root___eval_phase__act(vlSelf)) {
                vlSelf->__VactContinue = 1U;
            }
        }
        if (Vbug___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vbug___024root___eval_debug_assertions(Vbug___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vbug__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbug___024root___eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((vlSelf->a & 0xfeU))) {
        Verilated::overWidthError("a");}
    if (VL_UNLIKELY((vlSelf->c & 0xfeU))) {
        Verilated::overWidthError("c");}
    if (VL_UNLIKELY((vlSelf->clk & 0xfeU))) {
        Verilated::overWidthError("clk");}
}
#endif  // VL_DEBUG
