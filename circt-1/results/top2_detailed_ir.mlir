// -----// IR Dump Before CSE (cse) //----- //
moore.module @top_arc(in %clkin_data : !moore.l64, in %in_data : !moore.l192, out out_data : !moore.l192) {
  %_00_ = moore.variable : <l6>
  %clkin_data_0 = moore.net name "clkin_data" wire : <l64>
  %in_data_1 = moore.net name "in_data" wire : <l192>
  %out_data = moore.net wire : <l192>
  %0 = moore.read %clkin_data_0 : <l64>
  %1 = moore.extract %0 from 0 : l64 -> l1
  %clkin_0 = moore.net wire %1 : <l1>
  %2 = moore.read %clkin_data_0 : <l64>
  %3 = moore.extract %2 from 32 : l64 -> l1
  %rst = moore.net wire %3 : <l1>
  moore.procedure always_ff {
    moore.wait_event {
      %17 = moore.read %clkin_0 : <l1>
      moore.detect_event posedge %17 : l1
    }
    %10 = moore.read %rst : <l1>
    %11 = moore.not %10 : l1
    %12 = moore.to_builtin_bool %11 : l1
    cf.cond_br %12, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %13 = moore.constant 0 : i6
    %14 = moore.constant 0 : l6
    moore.nonblocking_assign %_00_, %14 : l6
    cf.br ^bb3
  ^bb2:  // pred: ^bb0
    %15 = moore.read %in_data_1 : <l192>
    %16 = moore.extract %15 from 2 : l192 -> l6
    moore.nonblocking_assign %_00_, %16 : l6
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    moore.return
  }
  %4 = moore.extract_ref %out_data from 0 : <l192> -> <l6>
  %5 = moore.read %_00_ : <l6>
  moore.assign %4, %5 : l6
  %6 = moore.extract_ref %out_data from 6 : <l192> -> <l186>
  %7 = moore.constant 0 : i186
  %8 = moore.constant 0 : l186
  moore.assign %6, %8 : l186
  moore.assign %clkin_data_0, %clkin_data : l64
  moore.assign %in_data_1, %in_data : l192
  %9 = moore.read %out_data : <l192>
  moore.output %9 : !moore.l192
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
moore.module @top_arc(in %clkin_data : !moore.l64, in %in_data : !moore.l192, out out_data : !moore.l192) {
  %_00_ = moore.variable : <l6>
  %clkin_data_0 = moore.net name "clkin_data" wire : <l64>
  %in_data_1 = moore.net name "in_data" wire : <l192>
  %out_data = moore.net wire : <l192>
  %0 = moore.read %clkin_data_0 : <l64>
  %1 = moore.extract %0 from 0 : l64 -> l1
  %clkin_0 = moore.net wire %1 : <l1>
  %2 = moore.extract %0 from 32 : l64 -> l1
  %rst = moore.net wire %2 : <l1>
  moore.procedure always_ff {
    moore.wait_event {
      %14 = moore.read %clkin_0 : <l1>
      moore.detect_event posedge %14 : l1
    }
    %8 = moore.read %rst : <l1>
    %9 = moore.not %8 : l1
    %10 = moore.to_builtin_bool %9 : l1
    cf.cond_br %10, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %11 = moore.constant 0 : l6
    moore.nonblocking_assign %_00_, %11 : l6
    cf.br ^bb3
  ^bb2:  // pred: ^bb0
    %12 = moore.read %in_data_1 : <l192>
    %13 = moore.extract %12 from 2 : l192 -> l6
    moore.nonblocking_assign %_00_, %13 : l6
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    moore.return
  }
  %3 = moore.extract_ref %out_data from 0 : <l192> -> <l6>
  %4 = moore.read %_00_ : <l6>
  moore.assign %3, %4 : l6
  %5 = moore.extract_ref %out_data from 6 : <l192> -> <l186>
  %6 = moore.constant 0 : l186
  moore.assign %5, %6 : l186
  moore.assign %clkin_data_0, %clkin_data : l64
  moore.assign %in_data_1, %in_data : l192
  %7 = moore.read %out_data : <l192>
  moore.output %7 : !moore.l192
}

// -----// IR Dump Before CreateVTables (moore-create-vtables) //----- //
module {
  moore.module @top_arc(in %clkin_data : !moore.l64, in %in_data : !moore.l192, out out_data : !moore.l192) {
    %0 = moore.constant 0 : l186
    %1 = moore.constant 0 : l6
    %_00_ = moore.variable : <l6>
    %in_data_0 = moore.net name "in_data" wire : <l192>
    %out_data = moore.net wire : <l192>
    %2 = moore.extract %clkin_data from 0 : l64 -> l1
    %clkin_0 = moore.net wire %2 : <l1>
    %3 = moore.extract %clkin_data from 32 : l64 -> l1
    %rst = moore.net wire %3 : <l1>
    moore.procedure always_ff {
      moore.wait_event {
        %13 = moore.read %clkin_0 : <l1>
        moore.detect_event posedge %13 : l1
      }
      %8 = moore.read %rst : <l1>
      %9 = moore.not %8 : l1
      %10 = moore.to_builtin_bool %9 : l1
      cf.cond_br %10, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      moore.nonblocking_assign %_00_, %1 : l6
      cf.br ^bb3
    ^bb2:  // pred: ^bb0
      %11 = moore.read %in_data_0 : <l192>
      %12 = moore.extract %11 from 2 : l192 -> l6
      moore.nonblocking_assign %_00_, %12 : l6
      cf.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      moore.return
    }
    %4 = moore.extract_ref %out_data from 0 : <l192> -> <l6>
    %5 = moore.read %_00_ : <l6>
    moore.assign %4, %5 : l6
    %6 = moore.extract_ref %out_data from 6 : <l192> -> <l186>
    moore.assign %6, %0 : l186
    moore.assign %in_data_0, %in_data : l192
    %7 = moore.read %out_data : <l192>
    moore.output %7 : !moore.l192
  }
}


// -----// IR Dump Before SymbolDCE (symbol-dce) //----- //
module {
  moore.module @top_arc(in %clkin_data : !moore.l64, in %in_data : !moore.l192, out out_data : !moore.l192) {
    %0 = moore.constant 0 : l186
    %1 = moore.constant 0 : l6
    %_00_ = moore.variable : <l6>
    %in_data_0 = moore.net name "in_data" wire : <l192>
    %out_data = moore.net wire : <l192>
    %2 = moore.extract %clkin_data from 0 : l64 -> l1
    %clkin_0 = moore.net wire %2 : <l1>
    %3 = moore.extract %clkin_data from 32 : l64 -> l1
    %rst = moore.net wire %3 : <l1>
    moore.procedure always_ff {
      moore.wait_event {
        %13 = moore.read %clkin_0 : <l1>
        moore.detect_event posedge %13 : l1
      }
      %8 = moore.read %rst : <l1>
      %9 = moore.not %8 : l1
      %10 = moore.to_builtin_bool %9 : l1
      cf.cond_br %10, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      moore.nonblocking_assign %_00_, %1 : l6
      cf.br ^bb3
    ^bb2:  // pred: ^bb0
      %11 = moore.read %in_data_0 : <l192>
      %12 = moore.extract %11 from 2 : l192 -> l6
      moore.nonblocking_assign %_00_, %12 : l6
      cf.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      moore.return
    }
    %4 = moore.extract_ref %out_data from 0 : <l192> -> <l6>
    %5 = moore.read %_00_ : <l6>
    moore.assign %4, %5 : l6
    %6 = moore.extract_ref %out_data from 6 : <l192> -> <l186>
    moore.assign %6, %0 : l186
    moore.assign %in_data_0, %in_data : l192
    %7 = moore.read %out_data : <l192>
    moore.output %7 : !moore.l192
  }
}


// -----// IR Dump Before LowerConcatRef (moore-lower-concatref) //----- //
moore.module @top_arc(in %clkin_data : !moore.l64, in %in_data : !moore.l192, out out_data : !moore.l192) {
  %0 = moore.constant 0 : l186
  %1 = moore.constant 0 : l6
  %_00_ = moore.variable : <l6>
  %in_data_0 = moore.net name "in_data" wire : <l192>
  %out_data = moore.net wire : <l192>
  %2 = moore.extract %clkin_data from 0 : l64 -> l1
  %clkin_0 = moore.net wire %2 : <l1>
  %3 = moore.extract %clkin_data from 32 : l64 -> l1
  %rst = moore.net wire %3 : <l1>
  moore.procedure always_ff {
    moore.wait_event {
      %13 = moore.read %clkin_0 : <l1>
      moore.detect_event posedge %13 : l1
    }
    %8 = moore.read %rst : <l1>
    %9 = moore.not %8 : l1
    %10 = moore.to_builtin_bool %9 : l1
    cf.cond_br %10, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    moore.nonblocking_assign %_00_, %1 : l6
    cf.br ^bb3
  ^bb2:  // pred: ^bb0
    %11 = moore.read %in_data_0 : <l192>
    %12 = moore.extract %11 from 2 : l192 -> l6
    moore.nonblocking_assign %_00_, %12 : l6
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    moore.return
  }
  %4 = moore.extract_ref %out_data from 0 : <l192> -> <l6>
  %5 = moore.read %_00_ : <l6>
  moore.assign %4, %5 : l6
  %6 = moore.extract_ref %out_data from 6 : <l192> -> <l186>
  moore.assign %6, %0 : l186
  moore.assign %in_data_0, %in_data : l192
  %7 = moore.read %out_data : <l192>
  moore.output %7 : !moore.l192
}

// -----// IR Dump Before SROA (sroa) //----- //
moore.module @top_arc(in %clkin_data : !moore.l64, in %in_data : !moore.l192, out out_data : !moore.l192) {
  %0 = moore.constant 0 : l186
  %1 = moore.constant 0 : l6
  %_00_ = moore.variable : <l6>
  %in_data_0 = moore.net name "in_data" wire : <l192>
  %out_data = moore.net wire : <l192>
  %2 = moore.extract %clkin_data from 0 : l64 -> l1
  %clkin_0 = moore.net wire %2 : <l1>
  %3 = moore.extract %clkin_data from 32 : l64 -> l1
  %rst = moore.net wire %3 : <l1>
  moore.procedure always_ff {
    moore.wait_event {
      %13 = moore.read %clkin_0 : <l1>
      moore.detect_event posedge %13 : l1
    }
    %8 = moore.read %rst : <l1>
    %9 = moore.not %8 : l1
    %10 = moore.to_builtin_bool %9 : l1
    cf.cond_br %10, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    moore.nonblocking_assign %_00_, %1 : l6
    cf.br ^bb3
  ^bb2:  // pred: ^bb0
    %11 = moore.read %in_data_0 : <l192>
    %12 = moore.extract %11 from 2 : l192 -> l6
    moore.nonblocking_assign %_00_, %12 : l6
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    moore.return
  }
  %4 = moore.extract_ref %out_data from 0 : <l192> -> <l6>
  %5 = moore.read %_00_ : <l6>
  moore.assign %4, %5 : l6
  %6 = moore.extract_ref %out_data from 6 : <l192> -> <l186>
  moore.assign %6, %0 : l186
  moore.assign %in_data_0, %in_data : l192
  %7 = moore.read %out_data : <l192>
  moore.output %7 : !moore.l192
}

// -----// IR Dump Before Mem2Reg (mem2reg) //----- //
moore.module @top_arc(in %clkin_data : !moore.l64, in %in_data : !moore.l192, out out_data : !moore.l192) {
  %0 = moore.constant 0 : l186
  %1 = moore.constant 0 : l6
  %_00_ = moore.variable : <l6>
  %in_data_0 = moore.net name "in_data" wire : <l192>
  %out_data = moore.net wire : <l192>
  %2 = moore.extract %clkin_data from 0 : l64 -> l1
  %clkin_0 = moore.net wire %2 : <l1>
  %3 = moore.extract %clkin_data from 32 : l64 -> l1
  %rst = moore.net wire %3 : <l1>
  moore.procedure always_ff {
    moore.wait_event {
      %13 = moore.read %clkin_0 : <l1>
      moore.detect_event posedge %13 : l1
    }
    %8 = moore.read %rst : <l1>
    %9 = moore.not %8 : l1
    %10 = moore.to_builtin_bool %9 : l1
    cf.cond_br %10, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    moore.nonblocking_assign %_00_, %1 : l6
    cf.br ^bb3
  ^bb2:  // pred: ^bb0
    %11 = moore.read %in_data_0 : <l192>
    %12 = moore.extract %11 from 2 : l192 -> l6
    moore.nonblocking_assign %_00_, %12 : l6
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    moore.return
  }
  %4 = moore.extract_ref %out_data from 0 : <l192> -> <l6>
  %5 = moore.read %_00_ : <l6>
  moore.assign %4, %5 : l6
  %6 = moore.extract_ref %out_data from 6 : <l192> -> <l186>
  moore.assign %6, %0 : l186
  moore.assign %in_data_0, %in_data : l192
  %7 = moore.read %out_data : <l192>
  moore.output %7 : !moore.l192
}

// -----// IR Dump Before CSE (cse) //----- //
moore.module @top_arc(in %clkin_data : !moore.l64, in %in_data : !moore.l192, out out_data : !moore.l192) {
  %0 = moore.constant 0 : l186
  %1 = moore.constant 0 : l6
  %_00_ = moore.variable : <l6>
  %in_data_0 = moore.net name "in_data" wire : <l192>
  %out_data = moore.net wire : <l192>
  %2 = moore.extract %clkin_data from 0 : l64 -> l1
  %clkin_0 = moore.net wire %2 : <l1>
  %3 = moore.extract %clkin_data from 32 : l64 -> l1
  %rst = moore.net wire %3 : <l1>
  moore.procedure always_ff {
    moore.wait_event {
      %13 = moore.read %clkin_0 : <l1>
      moore.detect_event posedge %13 : l1
    }
    %8 = moore.read %rst : <l1>
    %9 = moore.not %8 : l1
    %10 = moore.to_builtin_bool %9 : l1
    cf.cond_br %10, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    moore.nonblocking_assign %_00_, %1 : l6
    cf.br ^bb3
  ^bb2:  // pred: ^bb0
    %11 = moore.read %in_data_0 : <l192>
    %12 = moore.extract %11 from 2 : l192 -> l6
    moore.nonblocking_assign %_00_, %12 : l6
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    moore.return
  }
  %4 = moore.extract_ref %out_data from 0 : <l192> -> <l6>
  %5 = moore.read %_00_ : <l6>
  moore.assign %4, %5 : l6
  %6 = moore.extract_ref %out_data from 6 : <l192> -> <l186>
  moore.assign %6, %0 : l186
  moore.assign %in_data_0, %in_data : l192
  %7 = moore.read %out_data : <l192>
  moore.output %7 : !moore.l192
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
moore.module @top_arc(in %clkin_data : !moore.l64, in %in_data : !moore.l192, out out_data : !moore.l192) {
  %0 = moore.constant 0 : l186
  %1 = moore.constant 0 : l6
  %_00_ = moore.variable : <l6>
  %in_data_0 = moore.net name "in_data" wire : <l192>
  %out_data = moore.net wire : <l192>
  %2 = moore.extract %clkin_data from 0 : l64 -> l1
  %clkin_0 = moore.net wire %2 : <l1>
  %3 = moore.extract %clkin_data from 32 : l64 -> l1
  %rst = moore.net wire %3 : <l1>
  moore.procedure always_ff {
    moore.wait_event {
      %13 = moore.read %clkin_0 : <l1>
      moore.detect_event posedge %13 : l1
    }
    %8 = moore.read %rst : <l1>
    %9 = moore.not %8 : l1
    %10 = moore.to_builtin_bool %9 : l1
    cf.cond_br %10, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    moore.nonblocking_assign %_00_, %1 : l6
    cf.br ^bb3
  ^bb2:  // pred: ^bb0
    %11 = moore.read %in_data_0 : <l192>
    %12 = moore.extract %11 from 2 : l192 -> l6
    moore.nonblocking_assign %_00_, %12 : l6
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    moore.return
  }
  %4 = moore.extract_ref %out_data from 0 : <l192> -> <l6>
  %5 = moore.read %_00_ : <l6>
  moore.assign %4, %5 : l6
  %6 = moore.extract_ref %out_data from 6 : <l192> -> <l186>
  moore.assign %6, %0 : l186
  moore.assign %in_data_0, %in_data : l192
  %7 = moore.read %out_data : <l192>
  moore.output %7 : !moore.l192
}

// -----// IR Dump Before ConvertMooreToCore (convert-moore-to-core) //----- //
module {
  moore.module @top_arc(in %clkin_data : !moore.l64, in %in_data : !moore.l192, out out_data : !moore.l192) {
    %0 = moore.constant 0 : l186
    %1 = moore.constant 0 : l6
    %_00_ = moore.variable : <l6>
    %in_data_0 = moore.net name "in_data" wire : <l192>
    %out_data = moore.net wire : <l192>
    %2 = moore.extract %clkin_data from 0 : l64 -> l1
    %clkin_0 = moore.net wire %2 : <l1>
    %3 = moore.extract %clkin_data from 32 : l64 -> l1
    %rst = moore.net wire %3 : <l1>
    moore.procedure always_ff {
      moore.wait_event {
        %13 = moore.read %clkin_0 : <l1>
        moore.detect_event posedge %13 : l1
      }
      %8 = moore.read %rst : <l1>
      %9 = moore.not %8 : l1
      %10 = moore.to_builtin_bool %9 : l1
      cf.cond_br %10, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      moore.nonblocking_assign %_00_, %1 : l6
      cf.br ^bb3
    ^bb2:  // pred: ^bb0
      %11 = moore.read %in_data_0 : <l192>
      %12 = moore.extract %11 from 2 : l192 -> l6
      moore.nonblocking_assign %_00_, %12 : l6
      cf.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      moore.return
    }
    %4 = moore.extract_ref %out_data from 0 : <l192> -> <l6>
    %5 = moore.read %_00_ : <l6>
    moore.assign %4, %5 : l6
    %6 = moore.extract_ref %out_data from 6 : <l192> -> <l186>
    moore.assign %6, %0 : l186
    moore.assign %in_data_0, %in_data : l192
    %7 = moore.read %out_data : <l192>
    moore.output %7 : !moore.l192
  }
}


// -----// IR Dump Before CSE (cse) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %c0_i6_0 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6_0 : i6
  %c0_i192 = hw.constant 0 : i192
  %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
  %c0_i192_2 = hw.constant 0 : i192
  %out_data = llhd.sig %c0_i192_2 : i192
  %0 = comb.extract %clkin_data from 0 : (i64) -> i1
  %false = hw.constant false
  %clkin_0 = llhd.sig %false : i1
  %1 = llhd.prb %clkin_0 : i1
  %2 = llhd.constant_time <0ns, 0d, 1e>
  llhd.drv %clkin_0, %0 after %2 : i1
  %3 = comb.extract %clkin_data from 32 : (i64) -> i1
  %false_3 = hw.constant false
  %rst = llhd.sig %false_3 : i1
  %4 = llhd.constant_time <0ns, 0d, 1e>
  llhd.drv %rst, %3 after %4 : i1
  llhd.process {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb7
    cf.br ^bb2
  ^bb2:  // 2 preds: ^bb1, ^bb3
    %12 = llhd.prb %clkin_0 : i1
    llhd.wait (%1 : i1), ^bb3
  ^bb3:  // pred: ^bb2
    %13 = llhd.prb %clkin_0 : i1
    %true = hw.constant true
    %14 = comb.xor bin %12, %true : i1
    %15 = comb.and bin %14, %13 : i1
    cf.cond_br %15, ^bb4, ^bb2
  ^bb4:  // pred: ^bb3
    %16 = llhd.prb %rst : i1
    %true_4 = hw.constant true
    %17 = comb.xor %16, %true_4 : i1
    cf.cond_br %17, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %18 = llhd.constant_time <0ns, 1d, 0e>
    llhd.drv %_00_, %c0_i6 after %18 : i6
    cf.br ^bb7
  ^bb6:  // pred: ^bb4
    %19 = llhd.prb %in_data_1 : i192
    %20 = comb.extract %19 from 2 : (i192) -> i6
    %21 = llhd.constant_time <0ns, 1d, 0e>
    llhd.drv %_00_, %20 after %21 : i6
    cf.br ^bb7
  ^bb7:  // 2 preds: ^bb5, ^bb6
    cf.br ^bb1
  }
  %c0_i8 = hw.constant 0 : i8
  %5 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %6 = llhd.prb %_00_ : i6
  %7 = llhd.constant_time <0ns, 0d, 1e>
  llhd.drv %5, %6 after %7 : i6
  %c6_i8 = hw.constant 6 : i8
  %8 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  %9 = llhd.constant_time <0ns, 0d, 1e>
  llhd.drv %8, %c0_i186 after %9 : i186
  %10 = llhd.constant_time <0ns, 0d, 1e>
  llhd.drv %in_data_1, %in_data after %10 : i192
  %11 = llhd.prb %out_data : i192
  hw.output %11 : i192
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %c0_i192 = hw.constant 0 : i192
  %in_data_0 = llhd.sig name "in_data" %c0_i192 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %0 = comb.extract %clkin_data from 0 : (i64) -> i1
  %false = hw.constant false
  %clkin_0 = llhd.sig %false : i1
  %1 = llhd.prb %clkin_0 : i1
  %2 = llhd.constant_time <0ns, 0d, 1e>
  llhd.drv %clkin_0, %0 after %2 : i1
  %3 = comb.extract %clkin_data from 32 : (i64) -> i1
  %rst = llhd.sig %false : i1
  llhd.drv %rst, %3 after %2 : i1
  llhd.process {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb7
    cf.br ^bb2
  ^bb2:  // 2 preds: ^bb1, ^bb3
    %8 = llhd.prb %clkin_0 : i1
    llhd.wait (%1 : i1), ^bb3
  ^bb3:  // pred: ^bb2
    %9 = llhd.prb %clkin_0 : i1
    %true = hw.constant true
    %10 = comb.xor bin %8, %true : i1
    %11 = comb.and bin %10, %9 : i1
    cf.cond_br %11, ^bb4, ^bb2
  ^bb4:  // pred: ^bb3
    %12 = llhd.prb %rst : i1
    %13 = comb.xor %12, %true : i1
    cf.cond_br %13, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %14 = llhd.constant_time <0ns, 1d, 0e>
    llhd.drv %_00_, %c0_i6 after %14 : i6
    cf.br ^bb7
  ^bb6:  // pred: ^bb4
    %15 = llhd.prb %in_data_0 : i192
    %16 = comb.extract %15 from 2 : (i192) -> i6
    %17 = llhd.constant_time <0ns, 1d, 0e>
    llhd.drv %_00_, %16 after %17 : i6
    cf.br ^bb7
  ^bb7:  // 2 preds: ^bb5, ^bb6
    cf.br ^bb1
  }
  %c0_i8 = hw.constant 0 : i8
  %4 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %5 = llhd.prb %_00_ : i6
  llhd.drv %4, %5 after %2 : i6
  %c6_i8 = hw.constant 6 : i8
  %6 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %6, %c0_i186 after %2 : i186
  llhd.drv %in_data_0, %in_data after %2 : i192
  %7 = llhd.prb %out_data : i192
  hw.output %7 : i192
}

// -----// IR Dump Before WrapProceduralOpsPass (llhd-wrap-procedural-ops) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %c6_i8 = hw.constant 6 : i8
  %c0_i8 = hw.constant 0 : i8
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %true = hw.constant true
  %1 = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %c0_i192 = hw.constant 0 : i192
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %in_data_0 = llhd.sig name "in_data" %c0_i192 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %2 = comb.extract %clkin_data from 0 : (i64) -> i1
  %clkin_0 = llhd.sig %false : i1
  %3 = llhd.prb %clkin_0 : i1
  llhd.drv %clkin_0, %2 after %1 : i1
  %4 = comb.extract %clkin_data from 32 : (i64) -> i1
  %rst = llhd.sig %false : i1
  llhd.drv %rst, %4 after %1 : i1
  llhd.process {
    cf.br ^bb1
  ^bb1:  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
    %9 = llhd.prb %clkin_0 : i1
    llhd.wait (%3 : i1), ^bb2
  ^bb2:  // pred: ^bb1
    %10 = llhd.prb %clkin_0 : i1
    %11 = comb.xor bin %9, %true : i1
    %12 = comb.and bin %11, %10 : i1
    cf.cond_br %12, ^bb3, ^bb1
  ^bb3:  // pred: ^bb2
    %13 = llhd.prb %rst : i1
    %14 = comb.xor %13, %true : i1
    cf.cond_br %14, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llhd.drv %_00_, %c0_i6 after %0 : i6
    cf.br ^bb1
  ^bb5:  // pred: ^bb3
    %15 = llhd.prb %in_data_0 : i192
    %16 = comb.extract %15 from 2 : (i192) -> i6
    llhd.drv %_00_, %16 after %0 : i6
    cf.br ^bb1
  }
  %5 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %6 = llhd.prb %_00_ : i6
  llhd.drv %5, %6 after %1 : i6
  %7 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %7, %c0_i186 after %1 : i186
  llhd.drv %in_data_0, %in_data after %1 : i192
  %8 = llhd.prb %out_data : i192
  hw.output %8 : i192
}

// -----// IR Dump Before SCFToControlFlowPass (convert-scf-to-cf) //----- //
module {
  hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
    %c6_i8 = hw.constant 6 : i8
    %c0_i8 = hw.constant 0 : i8
    %0 = llhd.constant_time <0ns, 1d, 0e>
    %true = hw.constant true
    %1 = llhd.constant_time <0ns, 0d, 1e>
    %false = hw.constant false
    %c0_i192 = hw.constant 0 : i192
    %c0_i186 = hw.constant 0 : i186
    %c0_i6 = hw.constant 0 : i6
    %_00_ = llhd.sig %c0_i6 : i6
    %in_data_0 = llhd.sig name "in_data" %c0_i192 : i192
    %out_data = llhd.sig %c0_i192 : i192
    %2 = comb.extract %clkin_data from 0 : (i64) -> i1
    %clkin_0 = llhd.sig %false : i1
    %3 = llhd.prb %clkin_0 : i1
    llhd.drv %clkin_0, %2 after %1 : i1
    %4 = comb.extract %clkin_data from 32 : (i64) -> i1
    %rst = llhd.sig %false : i1
    llhd.drv %rst, %4 after %1 : i1
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
      %9 = llhd.prb %clkin_0 : i1
      llhd.wait (%3 : i1), ^bb2
    ^bb2:  // pred: ^bb1
      %10 = llhd.prb %clkin_0 : i1
      %11 = comb.xor bin %9, %true : i1
      %12 = comb.and bin %11, %10 : i1
      cf.cond_br %12, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %13 = llhd.prb %rst : i1
      %14 = comb.xor %13, %true : i1
      cf.cond_br %14, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %_00_, %c0_i6 after %0 : i6
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %15 = llhd.prb %in_data_0 : i192
      %16 = comb.extract %15 from 2 : (i192) -> i6
      llhd.drv %_00_, %16 after %0 : i6
      cf.br ^bb1
    }
    %5 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
    %6 = llhd.prb %_00_ : i6
    llhd.drv %5, %6 after %1 : i6
    %7 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
    llhd.drv %7, %c0_i186 after %1 : i186
    llhd.drv %in_data_0, %in_data after %1 : i192
    %8 = llhd.prb %out_data : i192
    hw.output %8 : i192
  }
}


// -----// IR Dump Before InlineCallsPass (llhd-inline-calls) //----- //
module {
  hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
    %c6_i8 = hw.constant 6 : i8
    %c0_i8 = hw.constant 0 : i8
    %0 = llhd.constant_time <0ns, 1d, 0e>
    %true = hw.constant true
    %1 = llhd.constant_time <0ns, 0d, 1e>
    %false = hw.constant false
    %c0_i192 = hw.constant 0 : i192
    %c0_i186 = hw.constant 0 : i186
    %c0_i6 = hw.constant 0 : i6
    %_00_ = llhd.sig %c0_i6 : i6
    %in_data_0 = llhd.sig name "in_data" %c0_i192 : i192
    %out_data = llhd.sig %c0_i192 : i192
    %2 = comb.extract %clkin_data from 0 : (i64) -> i1
    %clkin_0 = llhd.sig %false : i1
    %3 = llhd.prb %clkin_0 : i1
    llhd.drv %clkin_0, %2 after %1 : i1
    %4 = comb.extract %clkin_data from 32 : (i64) -> i1
    %rst = llhd.sig %false : i1
    llhd.drv %rst, %4 after %1 : i1
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
      %9 = llhd.prb %clkin_0 : i1
      llhd.wait (%3 : i1), ^bb2
    ^bb2:  // pred: ^bb1
      %10 = llhd.prb %clkin_0 : i1
      %11 = comb.xor bin %9, %true : i1
      %12 = comb.and bin %11, %10 : i1
      cf.cond_br %12, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %13 = llhd.prb %rst : i1
      %14 = comb.xor %13, %true : i1
      cf.cond_br %14, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %_00_, %c0_i6 after %0 : i6
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %15 = llhd.prb %in_data_0 : i192
      %16 = comb.extract %15 from 2 : (i192) -> i6
      llhd.drv %_00_, %16 after %0 : i6
      cf.br ^bb1
    }
    %5 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
    %6 = llhd.prb %_00_ : i6
    llhd.drv %5, %6 after %1 : i6
    %7 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
    llhd.drv %7, %c0_i186 after %1 : i186
    llhd.drv %in_data_0, %in_data after %1 : i192
    %8 = llhd.prb %out_data : i192
    hw.output %8 : i192
  }
}


// -----// IR Dump Before SymbolDCE (symbol-dce) //----- //
module {
  hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
    %c6_i8 = hw.constant 6 : i8
    %c0_i8 = hw.constant 0 : i8
    %0 = llhd.constant_time <0ns, 1d, 0e>
    %true = hw.constant true
    %1 = llhd.constant_time <0ns, 0d, 1e>
    %false = hw.constant false
    %c0_i192 = hw.constant 0 : i192
    %c0_i186 = hw.constant 0 : i186
    %c0_i6 = hw.constant 0 : i6
    %_00_ = llhd.sig %c0_i6 : i6
    %in_data_0 = llhd.sig name "in_data" %c0_i192 : i192
    %out_data = llhd.sig %c0_i192 : i192
    %2 = comb.extract %clkin_data from 0 : (i64) -> i1
    %clkin_0 = llhd.sig %false : i1
    %3 = llhd.prb %clkin_0 : i1
    llhd.drv %clkin_0, %2 after %1 : i1
    %4 = comb.extract %clkin_data from 32 : (i64) -> i1
    %rst = llhd.sig %false : i1
    llhd.drv %rst, %4 after %1 : i1
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
      %9 = llhd.prb %clkin_0 : i1
      llhd.wait (%3 : i1), ^bb2
    ^bb2:  // pred: ^bb1
      %10 = llhd.prb %clkin_0 : i1
      %11 = comb.xor bin %9, %true : i1
      %12 = comb.and bin %11, %10 : i1
      cf.cond_br %12, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %13 = llhd.prb %rst : i1
      %14 = comb.xor %13, %true : i1
      cf.cond_br %14, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %_00_, %c0_i6 after %0 : i6
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %15 = llhd.prb %in_data_0 : i192
      %16 = comb.extract %15 from 2 : (i192) -> i6
      llhd.drv %_00_, %16 after %0 : i6
      cf.br ^bb1
    }
    %5 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
    %6 = llhd.prb %_00_ : i6
    llhd.drv %5, %6 after %1 : i6
    %7 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
    llhd.drv %7, %c0_i186 after %1 : i186
    llhd.drv %in_data_0, %in_data after %1 : i192
    %8 = llhd.prb %out_data : i192
    hw.output %8 : i192
  }
}


// -----// IR Dump Before Mem2RegPass (llhd-mem2reg) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %c6_i8 = hw.constant 6 : i8
  %c0_i8 = hw.constant 0 : i8
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %true = hw.constant true
  %1 = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %c0_i192 = hw.constant 0 : i192
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %in_data_0 = llhd.sig name "in_data" %c0_i192 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %2 = comb.extract %clkin_data from 0 : (i64) -> i1
  %clkin_0 = llhd.sig %false : i1
  %3 = llhd.prb %clkin_0 : i1
  llhd.drv %clkin_0, %2 after %1 : i1
  %4 = comb.extract %clkin_data from 32 : (i64) -> i1
  %rst = llhd.sig %false : i1
  llhd.drv %rst, %4 after %1 : i1
  llhd.process {
    cf.br ^bb1
  ^bb1:  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
    %9 = llhd.prb %clkin_0 : i1
    llhd.wait (%3 : i1), ^bb2
  ^bb2:  // pred: ^bb1
    %10 = llhd.prb %clkin_0 : i1
    %11 = comb.xor bin %9, %true : i1
    %12 = comb.and bin %11, %10 : i1
    cf.cond_br %12, ^bb3, ^bb1
  ^bb3:  // pred: ^bb2
    %13 = llhd.prb %rst : i1
    %14 = comb.xor %13, %true : i1
    cf.cond_br %14, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llhd.drv %_00_, %c0_i6 after %0 : i6
    cf.br ^bb1
  ^bb5:  // pred: ^bb3
    %15 = llhd.prb %in_data_0 : i192
    %16 = comb.extract %15 from 2 : (i192) -> i6
    llhd.drv %_00_, %16 after %0 : i6
    cf.br ^bb1
  }
  %5 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %6 = llhd.prb %_00_ : i6
  llhd.drv %5, %6 after %1 : i6
  %7 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %7, %c0_i186 after %1 : i186
  llhd.drv %in_data_0, %in_data after %1 : i192
  %8 = llhd.prb %out_data : i192
  hw.output %8 : i192
}

// -----// IR Dump Before HoistSignalsPass (llhd-hoist-signals) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %c0_i8 = hw.constant 0 : i8
  %true_0 = hw.constant true
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %c0_i192 = hw.constant 0 : i192
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %1 = comb.extract %clkin_data from 0 : (i64) -> i1
  %clkin_0 = llhd.sig %false : i1
  %2 = llhd.prb %clkin_0 : i1
  llhd.drv %clkin_0, %1 after %0 : i1
  %3 = comb.extract %clkin_data from 32 : (i64) -> i1
  %rst = llhd.sig %false : i1
  llhd.drv %rst, %3 after %0 : i1
  llhd.process {
    %8 = llhd.prb %clkin_0 : i1
    %c0_i6_2 = hw.constant 0 : i6
    %false_3 = hw.constant false
    cf.br ^bb1(%8, %c0_i6_2, %false_3 : i1, i6, i1)
  ^bb1(%9: i1, %10: i6, %11: i1):  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
    %12 = llhd.constant_time <0ns, 1d, 0e>
    llhd.drv %_00_, %10 after %12 if %11 : i6
    llhd.wait (%2 : i1), ^bb2(%9 : i1)
  ^bb2(%13: i1):  // pred: ^bb1
    %14 = llhd.prb %clkin_0 : i1
    %15 = llhd.prb %rst : i1
    %16 = llhd.prb %in_data_1 : i192
    %17 = comb.xor bin %13, %true_0 : i1
    %18 = comb.and bin %17, %14 : i1
    %c0_i6_4 = hw.constant 0 : i6
    %false_5 = hw.constant false
    cf.cond_br %18, ^bb3, ^bb1(%14, %c0_i6_4, %false_5 : i1, i6, i1)
  ^bb3:  // pred: ^bb2
    %19 = comb.xor %15, %true_0 : i1
    cf.cond_br %19, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    cf.br ^bb1(%14, %c0_i6, %true : i1, i6, i1)
  ^bb5:  // pred: ^bb3
    %true_6 = hw.constant true
    %20 = comb.extract %16 from 2 : (i192) -> i6
    cf.br ^bb1(%14, %20, %true_6 : i1, i6, i1)
  }
  %4 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %5 = llhd.prb %_00_ : i6
  llhd.drv %4, %5 after %0 : i6
  %6 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %6, %c0_i186 after %0 : i186
  llhd.drv %in_data_1, %in_data after %0 : i192
  %7 = llhd.prb %out_data : i192
  hw.output %7 : i192
}

// -----// IR Dump Before DeseqPass (llhd-deseq) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %c0_i8 = hw.constant 0 : i8
  %true_0 = hw.constant true
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %c0_i192 = hw.constant 0 : i192
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
  %1 = llhd.prb %in_data_1 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %2 = comb.extract %clkin_data from 0 : (i64) -> i1
  %clkin_0 = llhd.sig %false : i1
  %3 = llhd.prb %clkin_0 : i1
  llhd.drv %clkin_0, %2 after %0 : i1
  %4 = comb.extract %clkin_data from 32 : (i64) -> i1
  %rst = llhd.sig %false : i1
  %5 = llhd.prb %rst : i1
  llhd.drv %rst, %4 after %0 : i1
  %6 = llhd.constant_time <0ns, 1d, 0e>
  %7:2 = llhd.process -> i6, i1 {
    %c0_i6_2 = hw.constant 0 : i6
    %false_3 = hw.constant false
    cf.br ^bb1(%3, %c0_i6_2, %false_3 : i1, i6, i1)
  ^bb1(%12: i1, %13: i6, %14: i1):  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
    llhd.wait yield (%13, %14 : i6, i1), (%3 : i1), ^bb2(%12 : i1)
  ^bb2(%15: i1):  // pred: ^bb1
    %16 = comb.xor bin %15, %true_0 : i1
    %17 = comb.and bin %16, %3 : i1
    %c0_i6_4 = hw.constant 0 : i6
    %false_5 = hw.constant false
    cf.cond_br %17, ^bb3, ^bb1(%3, %c0_i6_4, %false_5 : i1, i6, i1)
  ^bb3:  // pred: ^bb2
    %18 = comb.xor %5, %true_0 : i1
    cf.cond_br %18, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    cf.br ^bb1(%3, %c0_i6, %true : i1, i6, i1)
  ^bb5:  // pred: ^bb3
    %true_6 = hw.constant true
    %19 = comb.extract %1 from 2 : (i192) -> i6
    cf.br ^bb1(%3, %19, %true_6 : i1, i6, i1)
  }
  llhd.drv %_00_, %7#0 after %6 if %7#1 : i6
  %8 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %9 = llhd.prb %_00_ : i6
  llhd.drv %8, %9 after %0 : i6
  %10 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %10, %c0_i186 after %0 : i186
  llhd.drv %in_data_1, %in_data after %0 : i192
  %11 = llhd.prb %out_data : i192
  hw.output %11 : i192
}

// -----// IR Dump Before LowerProcessesPass (llhd-lower-processes) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %c0_i8 = hw.constant 0 : i8
  %true_0 = hw.constant true
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %c0_i192 = hw.constant 0 : i192
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
  %1 = llhd.prb %in_data_1 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %2 = comb.extract %clkin_data from 0 : (i64) -> i1
  %clkin_0 = llhd.sig %false : i1
  %3 = llhd.prb %clkin_0 : i1
  llhd.drv %clkin_0, %2 after %0 : i1
  %4 = comb.extract %clkin_data from 32 : (i64) -> i1
  %rst = llhd.sig %false : i1
  %5 = llhd.prb %rst : i1
  llhd.drv %rst, %4 after %0 : i1
  %6 = llhd.constant_time <0ns, 1d, 0e>
  %7:2 = llhd.combinational -> i6, i1 {
    %true_3 = hw.constant true
    %false_4 = hw.constant false
    %c0_i6_5 = hw.constant 0 : i6
    cf.br ^bb1(%true_3, %c0_i6_5, %false_4 : i1, i6, i1)
  ^bb1(%15: i1, %16: i6, %17: i1):  // pred: ^bb0
    %c0_i6_6 = hw.constant 0 : i6
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    %18 = comb.xor %5, %true_0 : i1
    cf.cond_br %18, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    cf.br ^bb5(%true_3, %c0_i6, %true : i1, i6, i1)
  ^bb4:  // pred: ^bb2
    %19 = comb.extract %1 from 2 : (i192) -> i6
    cf.br ^bb5(%true_3, %19, %true_3 : i1, i6, i1)
  ^bb5(%20: i1, %21: i6, %22: i1):  // 2 preds: ^bb3, ^bb4
    llhd.yield %21, %22 : i6, i1
  }
  %8 = seq.to_clock %3
  %9 = comb.mux bin %7#1, %7#0, %_00__2 : i6
  %_00__2 = seq.firreg %9 clock %8 {name = "_00_"} : i6
  %10 = llhd.constant_time <0ns, 0d, 1e>
  llhd.drv %_00_, %_00__2 after %10 : i6
  %11 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %12 = llhd.prb %_00_ : i6
  llhd.drv %11, %12 after %0 : i6
  %13 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %13, %c0_i186 after %0 : i186
  llhd.drv %in_data_1, %in_data after %0 : i192
  %14 = llhd.prb %out_data : i192
  hw.output %14 : i192
}

// -----// IR Dump Before CSE (cse) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %c0_i8 = hw.constant 0 : i8
  %true_0 = hw.constant true
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %c0_i192 = hw.constant 0 : i192
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
  %1 = llhd.prb %in_data_1 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %2 = comb.extract %clkin_data from 0 : (i64) -> i1
  %clkin_0 = llhd.sig %false : i1
  %3 = llhd.prb %clkin_0 : i1
  llhd.drv %clkin_0, %2 after %0 : i1
  %4 = comb.extract %clkin_data from 32 : (i64) -> i1
  %rst = llhd.sig %false : i1
  %5 = llhd.prb %rst : i1
  llhd.drv %rst, %4 after %0 : i1
  %6 = llhd.constant_time <0ns, 1d, 0e>
  %7:2 = llhd.combinational -> i6, i1 {
    %true_3 = hw.constant true
    %false_4 = hw.constant false
    %c0_i6_5 = hw.constant 0 : i6
    cf.br ^bb1(%true_3, %c0_i6_5, %false_4 : i1, i6, i1)
  ^bb1(%15: i1, %16: i6, %17: i1):  // pred: ^bb0
    %c0_i6_6 = hw.constant 0 : i6
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    %18 = comb.xor %5, %true_0 : i1
    cf.cond_br %18, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    cf.br ^bb5(%true_3, %c0_i6, %true : i1, i6, i1)
  ^bb4:  // pred: ^bb2
    %19 = comb.extract %1 from 2 : (i192) -> i6
    cf.br ^bb5(%true_3, %19, %true_3 : i1, i6, i1)
  ^bb5(%20: i1, %21: i6, %22: i1):  // 2 preds: ^bb3, ^bb4
    llhd.yield %21, %22 : i6, i1
  }
  %8 = seq.to_clock %3
  %9 = comb.mux bin %7#1, %7#0, %_00__2 : i6
  %_00__2 = seq.firreg %9 clock %8 {name = "_00_"} : i6
  %10 = llhd.constant_time <0ns, 0d, 1e>
  llhd.drv %_00_, %_00__2 after %10 : i6
  %11 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %12 = llhd.prb %_00_ : i6
  llhd.drv %11, %12 after %0 : i6
  %13 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %13, %c0_i186 after %0 : i186
  llhd.drv %in_data_1, %in_data after %0 : i192
  %14 = llhd.prb %out_data : i192
  hw.output %14 : i192
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %c0_i8 = hw.constant 0 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %c0_i192 = hw.constant 0 : i192
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %in_data_0 = llhd.sig name "in_data" %c0_i192 : i192
  %1 = llhd.prb %in_data_0 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %2 = comb.extract %clkin_data from 0 : (i64) -> i1
  %clkin_0 = llhd.sig %false : i1
  %3 = llhd.prb %clkin_0 : i1
  llhd.drv %clkin_0, %2 after %0 : i1
  %4 = comb.extract %clkin_data from 32 : (i64) -> i1
  %rst = llhd.sig %false : i1
  %5 = llhd.prb %rst : i1
  llhd.drv %rst, %4 after %0 : i1
  %6:2 = llhd.combinational -> i6, i1 {
    cf.br ^bb1(%true, %c0_i6, %false : i1, i6, i1)
  ^bb1(%13: i1, %14: i6, %15: i1):  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    %16 = comb.xor %5, %true : i1
    cf.cond_br %16, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    cf.br ^bb5(%true, %c0_i6, %true : i1, i6, i1)
  ^bb4:  // pred: ^bb2
    %17 = comb.extract %1 from 2 : (i192) -> i6
    cf.br ^bb5(%true, %17, %true : i1, i6, i1)
  ^bb5(%18: i1, %19: i6, %20: i1):  // 2 preds: ^bb3, ^bb4
    llhd.yield %19, %20 : i6, i1
  }
  %7 = seq.to_clock %3
  %8 = comb.mux bin %6#1, %6#0, %_00__1 : i6
  %_00__1 = seq.firreg %8 clock %7 {name = "_00_"} : i6
  llhd.drv %_00_, %_00__1 after %0 : i6
  %9 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %10 = llhd.prb %_00_ : i6
  llhd.drv %9, %10 after %0 : i6
  %11 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %11, %c0_i186 after %0 : i186
  llhd.drv %in_data_0, %in_data after %0 : i192
  %12 = llhd.prb %out_data : i192
  hw.output %12 : i192
}

// -----// IR Dump Before UnrollLoopsPass (llhd-unroll-loops) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %c0_i8 = hw.constant 0 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %c0_i192 = hw.constant 0 : i192
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %in_data_0 = llhd.sig name "in_data" %c0_i192 : i192
  %1 = llhd.prb %in_data_0 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %2 = comb.extract %clkin_data from 0 : (i64) -> i1
  %clkin_0 = llhd.sig %false : i1
  %3 = llhd.prb %clkin_0 : i1
  llhd.drv %clkin_0, %2 after %0 : i1
  %4 = comb.extract %clkin_data from 32 : (i64) -> i1
  %rst = llhd.sig %false : i1
  %5 = llhd.prb %rst : i1
  llhd.drv %rst, %4 after %0 : i1
  %6:2 = llhd.combinational -> i6, i1 {
    %13 = comb.xor %5, %true : i1
    cf.cond_br %13, ^bb2(%c0_i6, %true : i6, i1), ^bb1
  ^bb1:  // pred: ^bb0
    %14 = comb.extract %1 from 2 : (i192) -> i6
    cf.br ^bb2(%14, %true : i6, i1)
  ^bb2(%15: i6, %16: i1):  // 2 preds: ^bb0, ^bb1
    llhd.yield %15, %16 : i6, i1
  }
  %7 = seq.to_clock %3
  %8 = comb.mux bin %6#1, %6#0, %_00__1 : i6
  %_00__1 = seq.firreg %8 clock %7 {name = "_00_"} : i6
  llhd.drv %_00_, %_00__1 after %0 : i6
  %9 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %10 = llhd.prb %_00_ : i6
  llhd.drv %9, %10 after %0 : i6
  %11 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %11, %c0_i186 after %0 : i186
  llhd.drv %in_data_0, %in_data after %0 : i192
  %12 = llhd.prb %out_data : i192
  hw.output %12 : i192
}

// -----// IR Dump Before CSE (cse) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %c0_i8 = hw.constant 0 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %c0_i192 = hw.constant 0 : i192
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %in_data_0 = llhd.sig name "in_data" %c0_i192 : i192
  %1 = llhd.prb %in_data_0 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %2 = comb.extract %clkin_data from 0 : (i64) -> i1
  %clkin_0 = llhd.sig %false : i1
  %3 = llhd.prb %clkin_0 : i1
  llhd.drv %clkin_0, %2 after %0 : i1
  %4 = comb.extract %clkin_data from 32 : (i64) -> i1
  %rst = llhd.sig %false : i1
  %5 = llhd.prb %rst : i1
  llhd.drv %rst, %4 after %0 : i1
  %6:2 = llhd.combinational -> i6, i1 {
    %13 = comb.xor %5, %true : i1
    cf.cond_br %13, ^bb2(%c0_i6, %true : i6, i1), ^bb1
  ^bb1:  // pred: ^bb0
    %14 = comb.extract %1 from 2 : (i192) -> i6
    cf.br ^bb2(%14, %true : i6, i1)
  ^bb2(%15: i6, %16: i1):  // 2 preds: ^bb0, ^bb1
    llhd.yield %15, %16 : i6, i1
  }
  %7 = seq.to_clock %3
  %8 = comb.mux bin %6#1, %6#0, %_00__1 : i6
  %_00__1 = seq.firreg %8 clock %7 {name = "_00_"} : i6
  llhd.drv %_00_, %_00__1 after %0 : i6
  %9 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %10 = llhd.prb %_00_ : i6
  llhd.drv %9, %10 after %0 : i6
  %11 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %11, %c0_i186 after %0 : i186
  llhd.drv %in_data_0, %in_data after %0 : i192
  %12 = llhd.prb %out_data : i192
  hw.output %12 : i192
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %c0_i8 = hw.constant 0 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %c0_i192 = hw.constant 0 : i192
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %in_data_0 = llhd.sig name "in_data" %c0_i192 : i192
  %1 = llhd.prb %in_data_0 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %2 = comb.extract %clkin_data from 0 : (i64) -> i1
  %clkin_0 = llhd.sig %false : i1
  %3 = llhd.prb %clkin_0 : i1
  llhd.drv %clkin_0, %2 after %0 : i1
  %4 = comb.extract %clkin_data from 32 : (i64) -> i1
  %rst = llhd.sig %false : i1
  %5 = llhd.prb %rst : i1
  llhd.drv %rst, %4 after %0 : i1
  %6:2 = llhd.combinational -> i6, i1 {
    %13 = comb.xor %5, %true : i1
    cf.cond_br %13, ^bb2(%c0_i6, %true : i6, i1), ^bb1
  ^bb1:  // pred: ^bb0
    %14 = comb.extract %1 from 2 : (i192) -> i6
    cf.br ^bb2(%14, %true : i6, i1)
  ^bb2(%15: i6, %16: i1):  // 2 preds: ^bb0, ^bb1
    llhd.yield %15, %16 : i6, i1
  }
  %7 = seq.to_clock %3
  %8 = comb.mux bin %6#1, %6#0, %_00__1 : i6
  %_00__1 = seq.firreg %8 clock %7 {name = "_00_"} : i6
  llhd.drv %_00_, %_00__1 after %0 : i6
  %9 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %10 = llhd.prb %_00_ : i6
  llhd.drv %9, %10 after %0 : i6
  %11 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %11, %c0_i186 after %0 : i186
  llhd.drv %in_data_0, %in_data after %0 : i192
  %12 = llhd.prb %out_data : i192
  hw.output %12 : i192
}

// -----// IR Dump Before RemoveControlFlowPass (llhd-remove-control-flow) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %c0_i8 = hw.constant 0 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %c0_i192 = hw.constant 0 : i192
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %in_data_0 = llhd.sig name "in_data" %c0_i192 : i192
  %1 = llhd.prb %in_data_0 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %2 = comb.extract %clkin_data from 0 : (i64) -> i1
  %clkin_0 = llhd.sig %false : i1
  %3 = llhd.prb %clkin_0 : i1
  llhd.drv %clkin_0, %2 after %0 : i1
  %4 = comb.extract %clkin_data from 32 : (i64) -> i1
  %rst = llhd.sig %false : i1
  %5 = llhd.prb %rst : i1
  llhd.drv %rst, %4 after %0 : i1
  %6:2 = llhd.combinational -> i6, i1 {
    %13 = comb.xor %5, %true : i1
    cf.cond_br %13, ^bb2(%c0_i6, %true : i6, i1), ^bb1
  ^bb1:  // pred: ^bb0
    %14 = comb.extract %1 from 2 : (i192) -> i6
    cf.br ^bb2(%14, %true : i6, i1)
  ^bb2(%15: i6, %16: i1):  // 2 preds: ^bb0, ^bb1
    llhd.yield %15, %16 : i6, i1
  }
  %7 = seq.to_clock %3
  %8 = comb.mux bin %6#1, %6#0, %_00__1 : i6
  %_00__1 = seq.firreg %8 clock %7 {name = "_00_"} : i6
  llhd.drv %_00_, %_00__1 after %0 : i6
  %9 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %10 = llhd.prb %_00_ : i6
  llhd.drv %9, %10 after %0 : i6
  %11 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %11, %c0_i186 after %0 : i186
  llhd.drv %in_data_0, %in_data after %0 : i192
  %12 = llhd.prb %out_data : i192
  hw.output %12 : i192
}

// -----// IR Dump Before CSE (cse) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %c0_i8 = hw.constant 0 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %c0_i192 = hw.constant 0 : i192
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %in_data_0 = llhd.sig name "in_data" %c0_i192 : i192
  %1 = llhd.prb %in_data_0 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %2 = comb.extract %clkin_data from 0 : (i64) -> i1
  %clkin_0 = llhd.sig %false : i1
  %3 = llhd.prb %clkin_0 : i1
  llhd.drv %clkin_0, %2 after %0 : i1
  %4 = comb.extract %clkin_data from 32 : (i64) -> i1
  %rst = llhd.sig %false : i1
  %5 = llhd.prb %rst : i1
  llhd.drv %rst, %4 after %0 : i1
  %6:2 = llhd.combinational -> i6, i1 {
    %13 = comb.xor %5, %true : i1
    %14 = comb.extract %1 from 2 : (i192) -> i6
    %true_2 = hw.constant true
    %15 = comb.mux %5, %14, %c0_i6 : i6
    llhd.yield %15, %true : i6, i1
  }
  %7 = seq.to_clock %3
  %8 = comb.mux bin %6#1, %6#0, %_00__1 : i6
  %_00__1 = seq.firreg %8 clock %7 {name = "_00_"} : i6
  llhd.drv %_00_, %_00__1 after %0 : i6
  %9 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %10 = llhd.prb %_00_ : i6
  llhd.drv %9, %10 after %0 : i6
  %11 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %11, %c0_i186 after %0 : i186
  llhd.drv %in_data_0, %in_data after %0 : i192
  %12 = llhd.prb %out_data : i192
  hw.output %12 : i192
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %c0_i8 = hw.constant 0 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %c0_i192 = hw.constant 0 : i192
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %in_data_0 = llhd.sig name "in_data" %c0_i192 : i192
  %1 = llhd.prb %in_data_0 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %2 = comb.extract %clkin_data from 0 : (i64) -> i1
  %clkin_0 = llhd.sig %false : i1
  %3 = llhd.prb %clkin_0 : i1
  llhd.drv %clkin_0, %2 after %0 : i1
  %4 = comb.extract %clkin_data from 32 : (i64) -> i1
  %rst = llhd.sig %false : i1
  %5 = llhd.prb %rst : i1
  llhd.drv %rst, %4 after %0 : i1
  %6:2 = llhd.combinational -> i6, i1 {
    %13 = comb.extract %1 from 2 : (i192) -> i6
    %14 = comb.mux %5, %13, %c0_i6 : i6
    llhd.yield %14, %true : i6, i1
  }
  %7 = seq.to_clock %3
  %8 = comb.mux bin %6#1, %6#0, %_00__1 : i6
  %_00__1 = seq.firreg %8 clock %7 {name = "_00_"} : i6
  llhd.drv %_00_, %_00__1 after %0 : i6
  %9 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %10 = llhd.prb %_00_ : i6
  llhd.drv %9, %10 after %0 : i6
  %11 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %11, %c0_i186 after %0 : i186
  llhd.drv %in_data_0, %in_data after %0 : i192
  %12 = llhd.prb %out_data : i192
  hw.output %12 : i192
}

// -----// IR Dump Before MapArithToCombPass (map-arith-to-comb) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %c6_i8 = hw.constant 6 : i8
  %c0_i8 = hw.constant 0 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %c0_i192 = hw.constant 0 : i192
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %in_data_0 = llhd.sig name "in_data" %c0_i192 : i192
  %1 = llhd.prb %in_data_0 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %2 = comb.extract %clkin_data from 0 : (i64) -> i1
  %clkin_0 = llhd.sig %false : i1
  %3 = llhd.prb %clkin_0 : i1
  llhd.drv %clkin_0, %2 after %0 : i1
  %4 = comb.extract %clkin_data from 32 : (i64) -> i1
  %rst = llhd.sig %false : i1
  %5 = llhd.prb %rst : i1
  llhd.drv %rst, %4 after %0 : i1
  %6 = comb.extract %1 from 2 : (i192) -> i6
  %7 = comb.mux %5, %6, %c0_i6 : i6
  %8 = seq.to_clock %3
  %_00__1 = seq.firreg %7 clock %8 {name = "_00_"} : i6
  llhd.drv %_00_, %_00__1 after %0 : i6
  %9 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %10 = llhd.prb %_00_ : i6
  llhd.drv %9, %10 after %0 : i6
  %11 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %11, %c0_i186 after %0 : i186
  llhd.drv %in_data_0, %in_data after %0 : i192
  %12 = llhd.prb %out_data : i192
  hw.output %12 : i192
}

// -----// IR Dump Before CombineDrivesPass (llhd-combine-drives) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %c6_i8 = hw.constant 6 : i8
  %c0_i8 = hw.constant 0 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %c0_i192 = hw.constant 0 : i192
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %in_data_0 = llhd.sig name "in_data" %c0_i192 : i192
  %1 = llhd.prb %in_data_0 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %2 = comb.extract %clkin_data from 0 : (i64) -> i1
  %clkin_0 = llhd.sig %false : i1
  %3 = llhd.prb %clkin_0 : i1
  llhd.drv %clkin_0, %2 after %0 : i1
  %4 = comb.extract %clkin_data from 32 : (i64) -> i1
  %rst = llhd.sig %false : i1
  %5 = llhd.prb %rst : i1
  llhd.drv %rst, %4 after %0 : i1
  %6 = comb.extract %1 from 2 : (i192) -> i6
  %7 = comb.mux %5, %6, %c0_i6 : i6
  %8 = seq.to_clock %3
  %_00__1 = seq.firreg %7 clock %8 {name = "_00_"} : i6
  llhd.drv %_00_, %_00__1 after %0 : i6
  %9 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %10 = llhd.prb %_00_ : i6
  llhd.drv %9, %10 after %0 : i6
  %11 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %11, %c0_i186 after %0 : i186
  llhd.drv %in_data_0, %in_data after %0 : i192
  %12 = llhd.prb %out_data : i192
  hw.output %12 : i192
}

// -----// IR Dump Before Sig2Reg (llhd-sig2reg) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %c0_i192 = hw.constant 0 : i192
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %in_data_0 = llhd.sig name "in_data" %c0_i192 : i192
  %1 = llhd.prb %in_data_0 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %2 = comb.concat %c0_i186, %10 : i186, i6
  llhd.drv %out_data, %2 after %0 : i192
  %3 = comb.extract %clkin_data from 0 : (i64) -> i1
  %clkin_0 = llhd.sig %false : i1
  %4 = llhd.prb %clkin_0 : i1
  llhd.drv %clkin_0, %3 after %0 : i1
  %5 = comb.extract %clkin_data from 32 : (i64) -> i1
  %rst = llhd.sig %false : i1
  %6 = llhd.prb %rst : i1
  llhd.drv %rst, %5 after %0 : i1
  %7 = comb.extract %1 from 2 : (i192) -> i6
  %8 = comb.mux %6, %7, %c0_i6 : i6
  %9 = seq.to_clock %4
  %_00__1 = seq.firreg %8 clock %9 {name = "_00_"} : i6
  llhd.drv %_00_, %_00__1 after %0 : i6
  %10 = llhd.prb %_00_ : i6
  llhd.drv %in_data_0, %in_data after %0 : i192
  %11 = llhd.prb %out_data : i192
  hw.output %11 : i192
}

// -----// IR Dump Before CSE (cse) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %c0_i192 = hw.constant 0 : i192
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %c-1_i6 = hw.constant -1 : i6
  %c0_i6_0 = hw.constant 0 : i6
  %c-1_i6_1 = hw.constant -1 : i6
  %c0_i6_2 = hw.constant 0 : i6
  %c0_i6_3 = hw.constant 0 : i6
  %c-1_i192 = hw.constant -1 : i192
  %c0_i192_4 = hw.constant 0 : i192
  %c-1_i192_5 = hw.constant -1 : i192
  %c0_i192_6 = hw.constant 0 : i192
  %c0_i192_7 = hw.constant 0 : i192
  %c-1_i192_8 = hw.constant -1 : i192
  %c0_i192_9 = hw.constant 0 : i192
  %c-1_i192_10 = hw.constant -1 : i192
  %c0_i192_11 = hw.constant 0 : i192
  %c0_i192_12 = hw.constant 0 : i192
  %1 = comb.concat %c0_i186, %_00_ : i186, i6
  %2 = comb.extract %clkin_data from 0 : (i64) -> i1
  %true = hw.constant true
  %false_13 = hw.constant false
  %true_14 = hw.constant true
  %false_15 = hw.constant false
  %false_16 = hw.constant false
  %3 = comb.extract %clkin_data from 32 : (i64) -> i1
  %true_17 = hw.constant true
  %false_18 = hw.constant false
  %true_19 = hw.constant true
  %false_20 = hw.constant false
  %false_21 = hw.constant false
  %4 = comb.extract %in_data from 2 : (i192) -> i6
  %5 = comb.mux %3, %4, %c0_i6 : i6
  %6 = seq.to_clock %2
  %_00_ = seq.firreg %5 clock %6 : i6
  hw.output %1 : i192
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %0 = comb.concat %c0_i186, %_00_ : i186, i6
  %1 = comb.extract %clkin_data from 0 : (i64) -> i1
  %2 = comb.extract %clkin_data from 32 : (i64) -> i1
  %3 = comb.extract %in_data from 2 : (i192) -> i6
  %4 = comb.mux %2, %3, %c0_i6 : i6
  %5 = seq.to_clock %1
  %_00_ = seq.firreg %4 clock %5 : i6
  hw.output %0 : i192
}

// -----// IR Dump Before RegOfVecToMem (seq-reg-of-vec-to-mem) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %0 = comb.concat %c0_i186, %_00_ : i186, i6
  %1 = comb.extract %clkin_data from 0 : (i64) -> i1
  %2 = comb.extract %clkin_data from 32 : (i64) -> i1
  %3 = comb.extract %in_data from 2 : (i192) -> i6
  %4 = comb.mux %2, %3, %c0_i6 : i6
  %5 = seq.to_clock %1
  %_00_ = seq.firreg %4 clock %5 : i6
  hw.output %0 : i192
}

// -----// IR Dump Before CSE (cse) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %0 = comb.concat %c0_i186, %_00_ : i186, i6
  %1 = comb.extract %clkin_data from 0 : (i64) -> i1
  %2 = comb.extract %clkin_data from 32 : (i64) -> i1
  %3 = comb.extract %in_data from 2 : (i192) -> i6
  %4 = comb.mux %2, %3, %c0_i6 : i6
  %5 = seq.to_clock %1
  %_00_ = seq.firreg %4 clock %5 : i6
  hw.output %0 : i192
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %0 = comb.concat %c0_i186, %_00_ : i186, i6
  %1 = comb.extract %clkin_data from 0 : (i64) -> i1
  %2 = comb.extract %clkin_data from 32 : (i64) -> i1
  %3 = comb.extract %in_data from 2 : (i192) -> i6
  %4 = comb.mux %2, %3, %c0_i6 : i6
  %5 = seq.to_clock %1
  %_00_ = seq.firreg %4 clock %5 : i6
  hw.output %0 : i192
}

module {
  hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
    %c0_i186 = hw.constant 0 : i186
    %c0_i6 = hw.constant 0 : i6
    %0 = comb.concat %c0_i186, %_00_ : i186, i6
    %1 = comb.extract %clkin_data from 0 : (i64) -> i1
    %2 = comb.extract %clkin_data from 32 : (i64) -> i1
    %3 = comb.extract %in_data from 2 : (i192) -> i6
    %4 = comb.mux %2, %3, %c0_i6 : i6
    %5 = seq.to_clock %1
    %_00_ = seq.firreg %4 clock %5 : i6
    hw.output %0 : i192
  }
}
