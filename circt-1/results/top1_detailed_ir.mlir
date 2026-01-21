// -----// IR Dump Before CSE (cse) //----- //
moore.module @top_arc(in %clkin_data : !moore.l64, in %in_data : !moore.l192, out out_data : !moore.l192) {
  %_00_ = moore.variable : <l6>
  %clkin_data_0 = moore.net name "clkin_data" wire : <l64>
  %in_data_1 = moore.net name "in_data" wire : <l192>
  %out_data = moore.net wire : <l192>
  moore.procedure always_ff {
    moore.wait_event {
      %14 = moore.read %clkin_data_0 : <l64>
      %15 = moore.extract %14 from 0 : l64 -> l1
      moore.detect_event posedge %15 : l1
    }
    %6 = moore.read %clkin_data_0 : <l64>
    %7 = moore.extract %6 from 32 : l64 -> l1
    %8 = moore.not %7 : l1
    %9 = moore.to_builtin_bool %8 : l1
    cf.cond_br %9, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %10 = moore.constant 0 : i6
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
  %0 = moore.extract_ref %out_data from 0 : <l192> -> <l6>
  %1 = moore.read %_00_ : <l6>
  moore.assign %0, %1 : l6
  %2 = moore.extract_ref %out_data from 6 : <l192> -> <l186>
  %3 = moore.constant 0 : i186
  %4 = moore.constant 0 : l186
  moore.assign %2, %4 : l186
  moore.assign %clkin_data_0, %clkin_data : l64
  moore.assign %in_data_1, %in_data : l192
  %5 = moore.read %out_data : <l192>
  moore.output %5 : !moore.l192
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
moore.module @top_arc(in %clkin_data : !moore.l64, in %in_data : !moore.l192, out out_data : !moore.l192) {
  %_00_ = moore.variable : <l6>
  %clkin_data_0 = moore.net name "clkin_data" wire : <l64>
  %in_data_1 = moore.net name "in_data" wire : <l192>
  %out_data = moore.net wire : <l192>
  moore.procedure always_ff {
    moore.wait_event {
      %12 = moore.read %clkin_data_0 : <l64>
      %13 = moore.extract %12 from 0 : l64 -> l1
      moore.detect_event posedge %13 : l1
    }
    %5 = moore.read %clkin_data_0 : <l64>
    %6 = moore.extract %5 from 32 : l64 -> l1
    %7 = moore.not %6 : l1
    %8 = moore.to_builtin_bool %7 : l1
    cf.cond_br %8, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %9 = moore.constant 0 : l6
    moore.nonblocking_assign %_00_, %9 : l6
    cf.br ^bb3
  ^bb2:  // pred: ^bb0
    %10 = moore.read %in_data_1 : <l192>
    %11 = moore.extract %10 from 2 : l192 -> l6
    moore.nonblocking_assign %_00_, %11 : l6
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    moore.return
  }
  %0 = moore.extract_ref %out_data from 0 : <l192> -> <l6>
  %1 = moore.read %_00_ : <l6>
  moore.assign %0, %1 : l6
  %2 = moore.extract_ref %out_data from 6 : <l192> -> <l186>
  %3 = moore.constant 0 : l186
  moore.assign %2, %3 : l186
  moore.assign %clkin_data_0, %clkin_data : l64
  moore.assign %in_data_1, %in_data : l192
  %4 = moore.read %out_data : <l192>
  moore.output %4 : !moore.l192
}

// -----// IR Dump Before CreateVTables (moore-create-vtables) //----- //
module {
  moore.module @top_arc(in %clkin_data : !moore.l64, in %in_data : !moore.l192, out out_data : !moore.l192) {
    %0 = moore.constant 0 : l186
    %1 = moore.constant 0 : l6
    %_00_ = moore.variable : <l6>
    %clkin_data_0 = moore.net name "clkin_data" wire : <l64>
    %in_data_1 = moore.net name "in_data" wire : <l192>
    %out_data = moore.net wire : <l192>
    moore.procedure always_ff {
      moore.wait_event {
        %12 = moore.read %clkin_data_0 : <l64>
        %13 = moore.extract %12 from 0 : l64 -> l1
        moore.detect_event posedge %13 : l1
      }
      %6 = moore.read %clkin_data_0 : <l64>
      %7 = moore.extract %6 from 32 : l64 -> l1
      %8 = moore.not %7 : l1
      %9 = moore.to_builtin_bool %8 : l1
      cf.cond_br %9, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      moore.nonblocking_assign %_00_, %1 : l6
      cf.br ^bb3
    ^bb2:  // pred: ^bb0
      %10 = moore.read %in_data_1 : <l192>
      %11 = moore.extract %10 from 2 : l192 -> l6
      moore.nonblocking_assign %_00_, %11 : l6
      cf.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      moore.return
    }
    %2 = moore.extract_ref %out_data from 0 : <l192> -> <l6>
    %3 = moore.read %_00_ : <l6>
    moore.assign %2, %3 : l6
    %4 = moore.extract_ref %out_data from 6 : <l192> -> <l186>
    moore.assign %4, %0 : l186
    moore.assign %clkin_data_0, %clkin_data : l64
    moore.assign %in_data_1, %in_data : l192
    %5 = moore.read %out_data : <l192>
    moore.output %5 : !moore.l192
  }
}


// -----// IR Dump Before SymbolDCE (symbol-dce) //----- //
module {
  moore.module @top_arc(in %clkin_data : !moore.l64, in %in_data : !moore.l192, out out_data : !moore.l192) {
    %0 = moore.constant 0 : l186
    %1 = moore.constant 0 : l6
    %_00_ = moore.variable : <l6>
    %clkin_data_0 = moore.net name "clkin_data" wire : <l64>
    %in_data_1 = moore.net name "in_data" wire : <l192>
    %out_data = moore.net wire : <l192>
    moore.procedure always_ff {
      moore.wait_event {
        %12 = moore.read %clkin_data_0 : <l64>
        %13 = moore.extract %12 from 0 : l64 -> l1
        moore.detect_event posedge %13 : l1
      }
      %6 = moore.read %clkin_data_0 : <l64>
      %7 = moore.extract %6 from 32 : l64 -> l1
      %8 = moore.not %7 : l1
      %9 = moore.to_builtin_bool %8 : l1
      cf.cond_br %9, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      moore.nonblocking_assign %_00_, %1 : l6
      cf.br ^bb3
    ^bb2:  // pred: ^bb0
      %10 = moore.read %in_data_1 : <l192>
      %11 = moore.extract %10 from 2 : l192 -> l6
      moore.nonblocking_assign %_00_, %11 : l6
      cf.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      moore.return
    }
    %2 = moore.extract_ref %out_data from 0 : <l192> -> <l6>
    %3 = moore.read %_00_ : <l6>
    moore.assign %2, %3 : l6
    %4 = moore.extract_ref %out_data from 6 : <l192> -> <l186>
    moore.assign %4, %0 : l186
    moore.assign %clkin_data_0, %clkin_data : l64
    moore.assign %in_data_1, %in_data : l192
    %5 = moore.read %out_data : <l192>
    moore.output %5 : !moore.l192
  }
}


// -----// IR Dump Before LowerConcatRef (moore-lower-concatref) //----- //
moore.module @top_arc(in %clkin_data : !moore.l64, in %in_data : !moore.l192, out out_data : !moore.l192) {
  %0 = moore.constant 0 : l186
  %1 = moore.constant 0 : l6
  %_00_ = moore.variable : <l6>
  %clkin_data_0 = moore.net name "clkin_data" wire : <l64>
  %in_data_1 = moore.net name "in_data" wire : <l192>
  %out_data = moore.net wire : <l192>
  moore.procedure always_ff {
    moore.wait_event {
      %12 = moore.read %clkin_data_0 : <l64>
      %13 = moore.extract %12 from 0 : l64 -> l1
      moore.detect_event posedge %13 : l1
    }
    %6 = moore.read %clkin_data_0 : <l64>
    %7 = moore.extract %6 from 32 : l64 -> l1
    %8 = moore.not %7 : l1
    %9 = moore.to_builtin_bool %8 : l1
    cf.cond_br %9, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    moore.nonblocking_assign %_00_, %1 : l6
    cf.br ^bb3
  ^bb2:  // pred: ^bb0
    %10 = moore.read %in_data_1 : <l192>
    %11 = moore.extract %10 from 2 : l192 -> l6
    moore.nonblocking_assign %_00_, %11 : l6
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    moore.return
  }
  %2 = moore.extract_ref %out_data from 0 : <l192> -> <l6>
  %3 = moore.read %_00_ : <l6>
  moore.assign %2, %3 : l6
  %4 = moore.extract_ref %out_data from 6 : <l192> -> <l186>
  moore.assign %4, %0 : l186
  moore.assign %clkin_data_0, %clkin_data : l64
  moore.assign %in_data_1, %in_data : l192
  %5 = moore.read %out_data : <l192>
  moore.output %5 : !moore.l192
}

// -----// IR Dump Before SROA (sroa) //----- //
moore.module @top_arc(in %clkin_data : !moore.l64, in %in_data : !moore.l192, out out_data : !moore.l192) {
  %0 = moore.constant 0 : l186
  %1 = moore.constant 0 : l6
  %_00_ = moore.variable : <l6>
  %clkin_data_0 = moore.net name "clkin_data" wire : <l64>
  %in_data_1 = moore.net name "in_data" wire : <l192>
  %out_data = moore.net wire : <l192>
  moore.procedure always_ff {
    moore.wait_event {
      %12 = moore.read %clkin_data_0 : <l64>
      %13 = moore.extract %12 from 0 : l64 -> l1
      moore.detect_event posedge %13 : l1
    }
    %6 = moore.read %clkin_data_0 : <l64>
    %7 = moore.extract %6 from 32 : l64 -> l1
    %8 = moore.not %7 : l1
    %9 = moore.to_builtin_bool %8 : l1
    cf.cond_br %9, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    moore.nonblocking_assign %_00_, %1 : l6
    cf.br ^bb3
  ^bb2:  // pred: ^bb0
    %10 = moore.read %in_data_1 : <l192>
    %11 = moore.extract %10 from 2 : l192 -> l6
    moore.nonblocking_assign %_00_, %11 : l6
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    moore.return
  }
  %2 = moore.extract_ref %out_data from 0 : <l192> -> <l6>
  %3 = moore.read %_00_ : <l6>
  moore.assign %2, %3 : l6
  %4 = moore.extract_ref %out_data from 6 : <l192> -> <l186>
  moore.assign %4, %0 : l186
  moore.assign %clkin_data_0, %clkin_data : l64
  moore.assign %in_data_1, %in_data : l192
  %5 = moore.read %out_data : <l192>
  moore.output %5 : !moore.l192
}

// -----// IR Dump Before Mem2Reg (mem2reg) //----- //
moore.module @top_arc(in %clkin_data : !moore.l64, in %in_data : !moore.l192, out out_data : !moore.l192) {
  %0 = moore.constant 0 : l186
  %1 = moore.constant 0 : l6
  %_00_ = moore.variable : <l6>
  %clkin_data_0 = moore.net name "clkin_data" wire : <l64>
  %in_data_1 = moore.net name "in_data" wire : <l192>
  %out_data = moore.net wire : <l192>
  moore.procedure always_ff {
    moore.wait_event {
      %12 = moore.read %clkin_data_0 : <l64>
      %13 = moore.extract %12 from 0 : l64 -> l1
      moore.detect_event posedge %13 : l1
    }
    %6 = moore.read %clkin_data_0 : <l64>
    %7 = moore.extract %6 from 32 : l64 -> l1
    %8 = moore.not %7 : l1
    %9 = moore.to_builtin_bool %8 : l1
    cf.cond_br %9, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    moore.nonblocking_assign %_00_, %1 : l6
    cf.br ^bb3
  ^bb2:  // pred: ^bb0
    %10 = moore.read %in_data_1 : <l192>
    %11 = moore.extract %10 from 2 : l192 -> l6
    moore.nonblocking_assign %_00_, %11 : l6
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    moore.return
  }
  %2 = moore.extract_ref %out_data from 0 : <l192> -> <l6>
  %3 = moore.read %_00_ : <l6>
  moore.assign %2, %3 : l6
  %4 = moore.extract_ref %out_data from 6 : <l192> -> <l186>
  moore.assign %4, %0 : l186
  moore.assign %clkin_data_0, %clkin_data : l64
  moore.assign %in_data_1, %in_data : l192
  %5 = moore.read %out_data : <l192>
  moore.output %5 : !moore.l192
}

// -----// IR Dump Before CSE (cse) //----- //
moore.module @top_arc(in %clkin_data : !moore.l64, in %in_data : !moore.l192, out out_data : !moore.l192) {
  %0 = moore.constant 0 : l186
  %1 = moore.constant 0 : l6
  %_00_ = moore.variable : <l6>
  %clkin_data_0 = moore.net name "clkin_data" wire : <l64>
  %in_data_1 = moore.net name "in_data" wire : <l192>
  %out_data = moore.net wire : <l192>
  moore.procedure always_ff {
    moore.wait_event {
      %12 = moore.read %clkin_data_0 : <l64>
      %13 = moore.extract %12 from 0 : l64 -> l1
      moore.detect_event posedge %13 : l1
    }
    %6 = moore.read %clkin_data_0 : <l64>
    %7 = moore.extract %6 from 32 : l64 -> l1
    %8 = moore.not %7 : l1
    %9 = moore.to_builtin_bool %8 : l1
    cf.cond_br %9, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    moore.nonblocking_assign %_00_, %1 : l6
    cf.br ^bb3
  ^bb2:  // pred: ^bb0
    %10 = moore.read %in_data_1 : <l192>
    %11 = moore.extract %10 from 2 : l192 -> l6
    moore.nonblocking_assign %_00_, %11 : l6
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    moore.return
  }
  %2 = moore.extract_ref %out_data from 0 : <l192> -> <l6>
  %3 = moore.read %_00_ : <l6>
  moore.assign %2, %3 : l6
  %4 = moore.extract_ref %out_data from 6 : <l192> -> <l186>
  moore.assign %4, %0 : l186
  moore.assign %clkin_data_0, %clkin_data : l64
  moore.assign %in_data_1, %in_data : l192
  %5 = moore.read %out_data : <l192>
  moore.output %5 : !moore.l192
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
moore.module @top_arc(in %clkin_data : !moore.l64, in %in_data : !moore.l192, out out_data : !moore.l192) {
  %0 = moore.constant 0 : l186
  %1 = moore.constant 0 : l6
  %_00_ = moore.variable : <l6>
  %clkin_data_0 = moore.net name "clkin_data" wire : <l64>
  %in_data_1 = moore.net name "in_data" wire : <l192>
  %out_data = moore.net wire : <l192>
  moore.procedure always_ff {
    moore.wait_event {
      %12 = moore.read %clkin_data_0 : <l64>
      %13 = moore.extract %12 from 0 : l64 -> l1
      moore.detect_event posedge %13 : l1
    }
    %6 = moore.read %clkin_data_0 : <l64>
    %7 = moore.extract %6 from 32 : l64 -> l1
    %8 = moore.not %7 : l1
    %9 = moore.to_builtin_bool %8 : l1
    cf.cond_br %9, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    moore.nonblocking_assign %_00_, %1 : l6
    cf.br ^bb3
  ^bb2:  // pred: ^bb0
    %10 = moore.read %in_data_1 : <l192>
    %11 = moore.extract %10 from 2 : l192 -> l6
    moore.nonblocking_assign %_00_, %11 : l6
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    moore.return
  }
  %2 = moore.extract_ref %out_data from 0 : <l192> -> <l6>
  %3 = moore.read %_00_ : <l6>
  moore.assign %2, %3 : l6
  %4 = moore.extract_ref %out_data from 6 : <l192> -> <l186>
  moore.assign %4, %0 : l186
  moore.assign %clkin_data_0, %clkin_data : l64
  moore.assign %in_data_1, %in_data : l192
  %5 = moore.read %out_data : <l192>
  moore.output %5 : !moore.l192
}

// -----// IR Dump Before ConvertMooreToCore (convert-moore-to-core) //----- //
module {
  moore.module @top_arc(in %clkin_data : !moore.l64, in %in_data : !moore.l192, out out_data : !moore.l192) {
    %0 = moore.constant 0 : l186
    %1 = moore.constant 0 : l6
    %_00_ = moore.variable : <l6>
    %clkin_data_0 = moore.net name "clkin_data" wire : <l64>
    %in_data_1 = moore.net name "in_data" wire : <l192>
    %out_data = moore.net wire : <l192>
    moore.procedure always_ff {
      moore.wait_event {
        %12 = moore.read %clkin_data_0 : <l64>
        %13 = moore.extract %12 from 0 : l64 -> l1
        moore.detect_event posedge %13 : l1
      }
      %6 = moore.read %clkin_data_0 : <l64>
      %7 = moore.extract %6 from 32 : l64 -> l1
      %8 = moore.not %7 : l1
      %9 = moore.to_builtin_bool %8 : l1
      cf.cond_br %9, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      moore.nonblocking_assign %_00_, %1 : l6
      cf.br ^bb3
    ^bb2:  // pred: ^bb0
      %10 = moore.read %in_data_1 : <l192>
      %11 = moore.extract %10 from 2 : l192 -> l6
      moore.nonblocking_assign %_00_, %11 : l6
      cf.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      moore.return
    }
    %2 = moore.extract_ref %out_data from 0 : <l192> -> <l6>
    %3 = moore.read %_00_ : <l6>
    moore.assign %2, %3 : l6
    %4 = moore.extract_ref %out_data from 6 : <l192> -> <l186>
    moore.assign %4, %0 : l186
    moore.assign %clkin_data_0, %clkin_data : l64
    moore.assign %in_data_1, %in_data : l192
    %5 = moore.read %out_data : <l192>
    moore.output %5 : !moore.l192
  }
}


// -----// IR Dump Before CSE (cse) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %c0_i6_0 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6_0 : i6
  %c0_i64 = hw.constant 0 : i64
  %clkin_data_1 = llhd.sig name "clkin_data" %c0_i64 : i64
  %0 = llhd.prb %clkin_data_1 : i64
  %c0_i192 = hw.constant 0 : i192
  %in_data_2 = llhd.sig name "in_data" %c0_i192 : i192
  %c0_i192_3 = hw.constant 0 : i192
  %out_data = llhd.sig %c0_i192_3 : i192
  llhd.process {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb7
    cf.br ^bb2
  ^bb2:  // 2 preds: ^bb1, ^bb3
    %9 = llhd.prb %clkin_data_1 : i64
    %10 = comb.extract %9 from 0 : (i64) -> i1
    llhd.wait (%0 : i64), ^bb3
  ^bb3:  // pred: ^bb2
    %11 = llhd.prb %clkin_data_1 : i64
    %12 = comb.extract %11 from 0 : (i64) -> i1
    %true = hw.constant true
    %13 = comb.xor bin %10, %true : i1
    %14 = comb.and bin %13, %12 : i1
    cf.cond_br %14, ^bb4, ^bb2
  ^bb4:  // pred: ^bb3
    %15 = llhd.prb %clkin_data_1 : i64
    %16 = comb.extract %15 from 32 : (i64) -> i1
    %true_4 = hw.constant true
    %17 = comb.xor %16, %true_4 : i1
    cf.cond_br %17, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %18 = llhd.constant_time <0ns, 1d, 0e>
    llhd.drv %_00_, %c0_i6 after %18 : i6
    cf.br ^bb7
  ^bb6:  // pred: ^bb4
    %19 = llhd.prb %in_data_2 : i192
    %20 = comb.extract %19 from 2 : (i192) -> i6
    %21 = llhd.constant_time <0ns, 1d, 0e>
    llhd.drv %_00_, %20 after %21 : i6
    cf.br ^bb7
  ^bb7:  // 2 preds: ^bb5, ^bb6
    cf.br ^bb1
  }
  %c0_i8 = hw.constant 0 : i8
  %1 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %2 = llhd.prb %_00_ : i6
  %3 = llhd.constant_time <0ns, 0d, 1e>
  llhd.drv %1, %2 after %3 : i6
  %c6_i8 = hw.constant 6 : i8
  %4 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  %5 = llhd.constant_time <0ns, 0d, 1e>
  llhd.drv %4, %c0_i186 after %5 : i186
  %6 = llhd.constant_time <0ns, 0d, 1e>
  llhd.drv %clkin_data_1, %clkin_data after %6 : i64
  %7 = llhd.constant_time <0ns, 0d, 1e>
  llhd.drv %in_data_2, %in_data after %7 : i192
  %8 = llhd.prb %out_data : i192
  hw.output %8 : i192
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %c0_i64 = hw.constant 0 : i64
  %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
  %0 = llhd.prb %clkin_data_0 : i64
  %c0_i192 = hw.constant 0 : i192
  %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
  %out_data = llhd.sig %c0_i192 : i192
  llhd.process {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb7
    cf.br ^bb2
  ^bb2:  // 2 preds: ^bb1, ^bb3
    %6 = llhd.prb %clkin_data_0 : i64
    %7 = comb.extract %6 from 0 : (i64) -> i1
    llhd.wait (%0 : i64), ^bb3
  ^bb3:  // pred: ^bb2
    %8 = llhd.prb %clkin_data_0 : i64
    %9 = comb.extract %8 from 0 : (i64) -> i1
    %true = hw.constant true
    %10 = comb.xor bin %7, %true : i1
    %11 = comb.and bin %10, %9 : i1
    cf.cond_br %11, ^bb4, ^bb2
  ^bb4:  // pred: ^bb3
    %12 = llhd.prb %clkin_data_0 : i64
    %13 = comb.extract %12 from 32 : (i64) -> i1
    %14 = comb.xor %13, %true : i1
    cf.cond_br %14, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %15 = llhd.constant_time <0ns, 1d, 0e>
    llhd.drv %_00_, %c0_i6 after %15 : i6
    cf.br ^bb7
  ^bb6:  // pred: ^bb4
    %16 = llhd.prb %in_data_1 : i192
    %17 = comb.extract %16 from 2 : (i192) -> i6
    %18 = llhd.constant_time <0ns, 1d, 0e>
    llhd.drv %_00_, %17 after %18 : i6
    cf.br ^bb7
  ^bb7:  // 2 preds: ^bb5, ^bb6
    cf.br ^bb1
  }
  %c0_i8 = hw.constant 0 : i8
  %1 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %2 = llhd.prb %_00_ : i6
  %3 = llhd.constant_time <0ns, 0d, 1e>
  llhd.drv %1, %2 after %3 : i6
  %c6_i8 = hw.constant 6 : i8
  %4 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %4, %c0_i186 after %3 : i186
  llhd.drv %clkin_data_0, %clkin_data after %3 : i64
  llhd.drv %in_data_1, %in_data after %3 : i192
  %5 = llhd.prb %out_data : i192
  hw.output %5 : i192
}

// -----// IR Dump Before WrapProceduralOpsPass (llhd-wrap-procedural-ops) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %c6_i8 = hw.constant 6 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %1 = llhd.constant_time <0ns, 1d, 0e>
  %true = hw.constant true
  %c0_i192 = hw.constant 0 : i192
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
  %2 = llhd.prb %clkin_data_0 : i64
  %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
  %out_data = llhd.sig %c0_i192 : i192
  llhd.process {
    cf.br ^bb1
  ^bb1:  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
    %7 = llhd.prb %clkin_data_0 : i64
    %8 = comb.extract %7 from 0 : (i64) -> i1
    llhd.wait (%2 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %9 = llhd.prb %clkin_data_0 : i64
    %10 = comb.extract %9 from 0 : (i64) -> i1
    %11 = comb.xor bin %8, %true : i1
    %12 = comb.and bin %11, %10 : i1
    cf.cond_br %12, ^bb3, ^bb1
  ^bb3:  // pred: ^bb2
    %13 = llhd.prb %clkin_data_0 : i64
    %14 = comb.extract %13 from 32 : (i64) -> i1
    %15 = comb.xor %14, %true : i1
    cf.cond_br %15, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llhd.drv %_00_, %c0_i6 after %1 : i6
    cf.br ^bb1
  ^bb5:  // pred: ^bb3
    %16 = llhd.prb %in_data_1 : i192
    %17 = comb.extract %16 from 2 : (i192) -> i6
    llhd.drv %_00_, %17 after %1 : i6
    cf.br ^bb1
  }
  %3 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %4 = llhd.prb %_00_ : i6
  llhd.drv %3, %4 after %0 : i6
  %5 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %5, %c0_i186 after %0 : i186
  llhd.drv %clkin_data_0, %clkin_data after %0 : i64
  llhd.drv %in_data_1, %in_data after %0 : i192
  %6 = llhd.prb %out_data : i192
  hw.output %6 : i192
}

// -----// IR Dump Before SCFToControlFlowPass (convert-scf-to-cf) //----- //
module {
  hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
    %c6_i8 = hw.constant 6 : i8
    %0 = llhd.constant_time <0ns, 0d, 1e>
    %c0_i8 = hw.constant 0 : i8
    %1 = llhd.constant_time <0ns, 1d, 0e>
    %true = hw.constant true
    %c0_i192 = hw.constant 0 : i192
    %c0_i64 = hw.constant 0 : i64
    %c0_i186 = hw.constant 0 : i186
    %c0_i6 = hw.constant 0 : i6
    %_00_ = llhd.sig %c0_i6 : i6
    %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
    %2 = llhd.prb %clkin_data_0 : i64
    %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
    %out_data = llhd.sig %c0_i192 : i192
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
      %7 = llhd.prb %clkin_data_0 : i64
      %8 = comb.extract %7 from 0 : (i64) -> i1
      llhd.wait (%2 : i64), ^bb2
    ^bb2:  // pred: ^bb1
      %9 = llhd.prb %clkin_data_0 : i64
      %10 = comb.extract %9 from 0 : (i64) -> i1
      %11 = comb.xor bin %8, %true : i1
      %12 = comb.and bin %11, %10 : i1
      cf.cond_br %12, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %13 = llhd.prb %clkin_data_0 : i64
      %14 = comb.extract %13 from 32 : (i64) -> i1
      %15 = comb.xor %14, %true : i1
      cf.cond_br %15, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %_00_, %c0_i6 after %1 : i6
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %16 = llhd.prb %in_data_1 : i192
      %17 = comb.extract %16 from 2 : (i192) -> i6
      llhd.drv %_00_, %17 after %1 : i6
      cf.br ^bb1
    }
    %3 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
    %4 = llhd.prb %_00_ : i6
    llhd.drv %3, %4 after %0 : i6
    %5 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
    llhd.drv %5, %c0_i186 after %0 : i186
    llhd.drv %clkin_data_0, %clkin_data after %0 : i64
    llhd.drv %in_data_1, %in_data after %0 : i192
    %6 = llhd.prb %out_data : i192
    hw.output %6 : i192
  }
}


// -----// IR Dump Before InlineCallsPass (llhd-inline-calls) //----- //
module {
  hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
    %c6_i8 = hw.constant 6 : i8
    %0 = llhd.constant_time <0ns, 0d, 1e>
    %c0_i8 = hw.constant 0 : i8
    %1 = llhd.constant_time <0ns, 1d, 0e>
    %true = hw.constant true
    %c0_i192 = hw.constant 0 : i192
    %c0_i64 = hw.constant 0 : i64
    %c0_i186 = hw.constant 0 : i186
    %c0_i6 = hw.constant 0 : i6
    %_00_ = llhd.sig %c0_i6 : i6
    %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
    %2 = llhd.prb %clkin_data_0 : i64
    %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
    %out_data = llhd.sig %c0_i192 : i192
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
      %7 = llhd.prb %clkin_data_0 : i64
      %8 = comb.extract %7 from 0 : (i64) -> i1
      llhd.wait (%2 : i64), ^bb2
    ^bb2:  // pred: ^bb1
      %9 = llhd.prb %clkin_data_0 : i64
      %10 = comb.extract %9 from 0 : (i64) -> i1
      %11 = comb.xor bin %8, %true : i1
      %12 = comb.and bin %11, %10 : i1
      cf.cond_br %12, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %13 = llhd.prb %clkin_data_0 : i64
      %14 = comb.extract %13 from 32 : (i64) -> i1
      %15 = comb.xor %14, %true : i1
      cf.cond_br %15, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %_00_, %c0_i6 after %1 : i6
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %16 = llhd.prb %in_data_1 : i192
      %17 = comb.extract %16 from 2 : (i192) -> i6
      llhd.drv %_00_, %17 after %1 : i6
      cf.br ^bb1
    }
    %3 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
    %4 = llhd.prb %_00_ : i6
    llhd.drv %3, %4 after %0 : i6
    %5 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
    llhd.drv %5, %c0_i186 after %0 : i186
    llhd.drv %clkin_data_0, %clkin_data after %0 : i64
    llhd.drv %in_data_1, %in_data after %0 : i192
    %6 = llhd.prb %out_data : i192
    hw.output %6 : i192
  }
}


// -----// IR Dump Before SymbolDCE (symbol-dce) //----- //
module {
  hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
    %c6_i8 = hw.constant 6 : i8
    %0 = llhd.constant_time <0ns, 0d, 1e>
    %c0_i8 = hw.constant 0 : i8
    %1 = llhd.constant_time <0ns, 1d, 0e>
    %true = hw.constant true
    %c0_i192 = hw.constant 0 : i192
    %c0_i64 = hw.constant 0 : i64
    %c0_i186 = hw.constant 0 : i186
    %c0_i6 = hw.constant 0 : i6
    %_00_ = llhd.sig %c0_i6 : i6
    %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
    %2 = llhd.prb %clkin_data_0 : i64
    %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
    %out_data = llhd.sig %c0_i192 : i192
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
      %7 = llhd.prb %clkin_data_0 : i64
      %8 = comb.extract %7 from 0 : (i64) -> i1
      llhd.wait (%2 : i64), ^bb2
    ^bb2:  // pred: ^bb1
      %9 = llhd.prb %clkin_data_0 : i64
      %10 = comb.extract %9 from 0 : (i64) -> i1
      %11 = comb.xor bin %8, %true : i1
      %12 = comb.and bin %11, %10 : i1
      cf.cond_br %12, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %13 = llhd.prb %clkin_data_0 : i64
      %14 = comb.extract %13 from 32 : (i64) -> i1
      %15 = comb.xor %14, %true : i1
      cf.cond_br %15, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %_00_, %c0_i6 after %1 : i6
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %16 = llhd.prb %in_data_1 : i192
      %17 = comb.extract %16 from 2 : (i192) -> i6
      llhd.drv %_00_, %17 after %1 : i6
      cf.br ^bb1
    }
    %3 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
    %4 = llhd.prb %_00_ : i6
    llhd.drv %3, %4 after %0 : i6
    %5 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
    llhd.drv %5, %c0_i186 after %0 : i186
    llhd.drv %clkin_data_0, %clkin_data after %0 : i64
    llhd.drv %in_data_1, %in_data after %0 : i192
    %6 = llhd.prb %out_data : i192
    hw.output %6 : i192
  }
}


// -----// IR Dump Before Mem2RegPass (llhd-mem2reg) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %c6_i8 = hw.constant 6 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %1 = llhd.constant_time <0ns, 1d, 0e>
  %true = hw.constant true
  %c0_i192 = hw.constant 0 : i192
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
  %2 = llhd.prb %clkin_data_0 : i64
  %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
  %out_data = llhd.sig %c0_i192 : i192
  llhd.process {
    cf.br ^bb1
  ^bb1:  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
    %7 = llhd.prb %clkin_data_0 : i64
    %8 = comb.extract %7 from 0 : (i64) -> i1
    llhd.wait (%2 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %9 = llhd.prb %clkin_data_0 : i64
    %10 = comb.extract %9 from 0 : (i64) -> i1
    %11 = comb.xor bin %8, %true : i1
    %12 = comb.and bin %11, %10 : i1
    cf.cond_br %12, ^bb3, ^bb1
  ^bb3:  // pred: ^bb2
    %13 = llhd.prb %clkin_data_0 : i64
    %14 = comb.extract %13 from 32 : (i64) -> i1
    %15 = comb.xor %14, %true : i1
    cf.cond_br %15, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llhd.drv %_00_, %c0_i6 after %1 : i6
    cf.br ^bb1
  ^bb5:  // pred: ^bb3
    %16 = llhd.prb %in_data_1 : i192
    %17 = comb.extract %16 from 2 : (i192) -> i6
    llhd.drv %_00_, %17 after %1 : i6
    cf.br ^bb1
  }
  %3 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %4 = llhd.prb %_00_ : i6
  llhd.drv %3, %4 after %0 : i6
  %5 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %5, %c0_i186 after %0 : i186
  llhd.drv %clkin_data_0, %clkin_data after %0 : i64
  llhd.drv %in_data_1, %in_data after %0 : i192
  %6 = llhd.prb %out_data : i192
  hw.output %6 : i192
}

// -----// IR Dump Before HoistSignalsPass (llhd-hoist-signals) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %true_0 = hw.constant true
  %c0_i192 = hw.constant 0 : i192
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_1 = llhd.sig name "clkin_data" %c0_i64 : i64
  %1 = llhd.prb %clkin_data_1 : i64
  %in_data_2 = llhd.sig name "in_data" %c0_i192 : i192
  %out_data = llhd.sig %c0_i192 : i192
  llhd.process {
    %6 = llhd.prb %clkin_data_1 : i64
    %c0_i6_3 = hw.constant 0 : i6
    %false = hw.constant false
    cf.br ^bb1(%6, %c0_i6_3, %false : i64, i6, i1)
  ^bb1(%7: i64, %8: i6, %9: i1):  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
    %10 = comb.extract %7 from 0 : (i64) -> i1
    %11 = llhd.constant_time <0ns, 1d, 0e>
    llhd.drv %_00_, %8 after %11 if %9 : i6
    llhd.wait (%1 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %12 = llhd.prb %clkin_data_1 : i64
    %13 = llhd.prb %in_data_2 : i192
    %14 = comb.extract %12 from 0 : (i64) -> i1
    %15 = comb.xor bin %10, %true_0 : i1
    %16 = comb.and bin %15, %14 : i1
    %c0_i6_4 = hw.constant 0 : i6
    %false_5 = hw.constant false
    cf.cond_br %16, ^bb3, ^bb1(%12, %c0_i6_4, %false_5 : i64, i6, i1)
  ^bb3:  // pred: ^bb2
    %17 = comb.extract %12 from 32 : (i64) -> i1
    %18 = comb.xor %17, %true_0 : i1
    cf.cond_br %18, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    cf.br ^bb1(%12, %c0_i6, %true : i64, i6, i1)
  ^bb5:  // pred: ^bb3
    %true_6 = hw.constant true
    %19 = comb.extract %13 from 2 : (i192) -> i6
    cf.br ^bb1(%12, %19, %true_6 : i64, i6, i1)
  }
  %2 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %3 = llhd.prb %_00_ : i6
  llhd.drv %2, %3 after %0 : i6
  %4 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %4, %c0_i186 after %0 : i186
  llhd.drv %clkin_data_1, %clkin_data after %0 : i64
  llhd.drv %in_data_2, %in_data after %0 : i192
  %5 = llhd.prb %out_data : i192
  hw.output %5 : i192
}

// -----// IR Dump Before DeseqPass (llhd-deseq) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %true_0 = hw.constant true
  %c0_i192 = hw.constant 0 : i192
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_1 = llhd.sig name "clkin_data" %c0_i64 : i64
  %1 = llhd.prb %clkin_data_1 : i64
  %in_data_2 = llhd.sig name "in_data" %c0_i192 : i192
  %2 = llhd.prb %in_data_2 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %3 = llhd.constant_time <0ns, 1d, 0e>
  %4:2 = llhd.process -> i6, i1 {
    %9 = llhd.prb %clkin_data_1 : i64
    %c0_i6_3 = hw.constant 0 : i6
    %false = hw.constant false
    cf.br ^bb1(%9, %c0_i6_3, %false : i64, i6, i1)
  ^bb1(%10: i64, %11: i6, %12: i1):  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
    %13 = comb.extract %10 from 0 : (i64) -> i1
    llhd.wait yield (%11, %12 : i6, i1), (%1 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %14 = llhd.prb %clkin_data_1 : i64
    %15 = comb.extract %14 from 0 : (i64) -> i1
    %16 = comb.xor bin %13, %true_0 : i1
    %17 = comb.and bin %16, %15 : i1
    %c0_i6_4 = hw.constant 0 : i6
    %false_5 = hw.constant false
    cf.cond_br %17, ^bb3, ^bb1(%14, %c0_i6_4, %false_5 : i64, i6, i1)
  ^bb3:  // pred: ^bb2
    %18 = comb.extract %14 from 32 : (i64) -> i1
    %19 = comb.xor %18, %true_0 : i1
    cf.cond_br %19, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    cf.br ^bb1(%14, %c0_i6, %true : i64, i6, i1)
  ^bb5:  // pred: ^bb3
    %true_6 = hw.constant true
    %20 = comb.extract %2 from 2 : (i192) -> i6
    cf.br ^bb1(%14, %20, %true_6 : i64, i6, i1)
  }
  llhd.drv %_00_, %4#0 after %3 if %4#1 : i6
  %5 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %6 = llhd.prb %_00_ : i6
  llhd.drv %5, %6 after %0 : i6
  %7 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %7, %c0_i186 after %0 : i186
  llhd.drv %clkin_data_1, %clkin_data after %0 : i64
  llhd.drv %in_data_2, %in_data after %0 : i192
  %8 = llhd.prb %out_data : i192
  hw.output %8 : i192
}

// -----// IR Dump Before LowerProcessesPass (llhd-lower-processes) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %true_0 = hw.constant true
  %c0_i192 = hw.constant 0 : i192
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_1 = llhd.sig name "clkin_data" %c0_i64 : i64
  %1 = llhd.prb %clkin_data_1 : i64
  %in_data_2 = llhd.sig name "in_data" %c0_i192 : i192
  %2 = llhd.prb %in_data_2 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %3 = llhd.constant_time <0ns, 1d, 0e>
  %4:2 = llhd.process -> i6, i1 {
    %9 = llhd.prb %clkin_data_1 : i64
    %c0_i6_3 = hw.constant 0 : i6
    %false = hw.constant false
    cf.br ^bb1(%9, %c0_i6_3, %false : i64, i6, i1)
  ^bb1(%10: i64, %11: i6, %12: i1):  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
    %13 = comb.extract %10 from 0 : (i64) -> i1
    llhd.wait yield (%11, %12 : i6, i1), (%1 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %14 = llhd.prb %clkin_data_1 : i64
    %15 = comb.extract %14 from 0 : (i64) -> i1
    %16 = comb.xor bin %13, %true_0 : i1
    %17 = comb.and bin %16, %15 : i1
    %c0_i6_4 = hw.constant 0 : i6
    %false_5 = hw.constant false
    cf.cond_br %17, ^bb3, ^bb1(%14, %c0_i6_4, %false_5 : i64, i6, i1)
  ^bb3:  // pred: ^bb2
    %18 = comb.extract %14 from 32 : (i64) -> i1
    %19 = comb.xor %18, %true_0 : i1
    cf.cond_br %19, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    cf.br ^bb1(%14, %c0_i6, %true : i64, i6, i1)
  ^bb5:  // pred: ^bb3
    %true_6 = hw.constant true
    %20 = comb.extract %2 from 2 : (i192) -> i6
    cf.br ^bb1(%14, %20, %true_6 : i64, i6, i1)
  }
  llhd.drv %_00_, %4#0 after %3 if %4#1 : i6
  %5 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %6 = llhd.prb %_00_ : i6
  llhd.drv %5, %6 after %0 : i6
  %7 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %7, %c0_i186 after %0 : i186
  llhd.drv %clkin_data_1, %clkin_data after %0 : i64
  llhd.drv %in_data_2, %in_data after %0 : i192
  %8 = llhd.prb %out_data : i192
  hw.output %8 : i192
}

// -----// IR Dump Before CSE (cse) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %true_0 = hw.constant true
  %c0_i192 = hw.constant 0 : i192
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_1 = llhd.sig name "clkin_data" %c0_i64 : i64
  %1 = llhd.prb %clkin_data_1 : i64
  %in_data_2 = llhd.sig name "in_data" %c0_i192 : i192
  %2 = llhd.prb %in_data_2 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %3 = llhd.constant_time <0ns, 1d, 0e>
  %4:2 = llhd.process -> i6, i1 {
    %9 = llhd.prb %clkin_data_1 : i64
    %c0_i6_3 = hw.constant 0 : i6
    %false = hw.constant false
    cf.br ^bb1(%9, %c0_i6_3, %false : i64, i6, i1)
  ^bb1(%10: i64, %11: i6, %12: i1):  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
    %13 = comb.extract %10 from 0 : (i64) -> i1
    llhd.wait yield (%11, %12 : i6, i1), (%1 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %14 = llhd.prb %clkin_data_1 : i64
    %15 = comb.extract %14 from 0 : (i64) -> i1
    %16 = comb.xor bin %13, %true_0 : i1
    %17 = comb.and bin %16, %15 : i1
    %c0_i6_4 = hw.constant 0 : i6
    %false_5 = hw.constant false
    cf.cond_br %17, ^bb3, ^bb1(%14, %c0_i6_4, %false_5 : i64, i6, i1)
  ^bb3:  // pred: ^bb2
    %18 = comb.extract %14 from 32 : (i64) -> i1
    %19 = comb.xor %18, %true_0 : i1
    cf.cond_br %19, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    cf.br ^bb1(%14, %c0_i6, %true : i64, i6, i1)
  ^bb5:  // pred: ^bb3
    %true_6 = hw.constant true
    %20 = comb.extract %2 from 2 : (i192) -> i6
    cf.br ^bb1(%14, %20, %true_6 : i64, i6, i1)
  }
  llhd.drv %_00_, %4#0 after %3 if %4#1 : i6
  %5 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %6 = llhd.prb %_00_ : i6
  llhd.drv %5, %6 after %0 : i6
  %7 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %7, %c0_i186 after %0 : i186
  llhd.drv %clkin_data_1, %clkin_data after %0 : i64
  llhd.drv %in_data_2, %in_data after %0 : i192
  %8 = llhd.prb %out_data : i192
  hw.output %8 : i192
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %c0_i192 = hw.constant 0 : i192
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
  %1 = llhd.prb %clkin_data_0 : i64
  %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
  %2 = llhd.prb %in_data_1 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %3 = llhd.constant_time <0ns, 1d, 0e>
  %4:2 = llhd.process -> i6, i1 {
    %9 = llhd.prb %clkin_data_0 : i64
    %false = hw.constant false
    cf.br ^bb1(%9, %c0_i6, %false : i64, i6, i1)
  ^bb1(%10: i64, %11: i6, %12: i1):  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
    %13 = comb.extract %10 from 0 : (i64) -> i1
    llhd.wait yield (%11, %12 : i6, i1), (%1 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %14 = llhd.prb %clkin_data_0 : i64
    %15 = comb.extract %14 from 0 : (i64) -> i1
    %16 = comb.xor bin %13, %true : i1
    %17 = comb.and bin %16, %15 : i1
    cf.cond_br %17, ^bb3, ^bb1(%14, %c0_i6, %false : i64, i6, i1)
  ^bb3:  // pred: ^bb2
    %18 = comb.extract %14 from 32 : (i64) -> i1
    %19 = comb.xor %18, %true : i1
    cf.cond_br %19, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    cf.br ^bb1(%14, %c0_i6, %true : i64, i6, i1)
  ^bb5:  // pred: ^bb3
    %20 = comb.extract %2 from 2 : (i192) -> i6
    cf.br ^bb1(%14, %20, %true : i64, i6, i1)
  }
  llhd.drv %_00_, %4#0 after %3 if %4#1 : i6
  %5 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %6 = llhd.prb %_00_ : i6
  llhd.drv %5, %6 after %0 : i6
  %7 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %7, %c0_i186 after %0 : i186
  llhd.drv %clkin_data_0, %clkin_data after %0 : i64
  llhd.drv %in_data_1, %in_data after %0 : i192
  %8 = llhd.prb %out_data : i192
  hw.output %8 : i192
}

// -----// IR Dump Before UnrollLoopsPass (llhd-unroll-loops) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %false = hw.constant false
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %1 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %c0_i192 = hw.constant 0 : i192
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
  %2 = llhd.prb %clkin_data_0 : i64
  %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
  %3 = llhd.prb %in_data_1 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %4:2 = llhd.process -> i6, i1 {
    %9 = llhd.prb %clkin_data_0 : i64
    cf.br ^bb1(%9, %c0_i6, %false : i64, i6, i1)
  ^bb1(%10: i64, %11: i6, %12: i1):  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb4
    %13 = comb.extract %10 from 0 : (i64) -> i1
    llhd.wait yield (%11, %12 : i6, i1), (%2 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %14 = llhd.prb %clkin_data_0 : i64
    %15 = comb.extract %14 from 0 : (i64) -> i1
    %16 = comb.xor bin %13, %true : i1
    %17 = comb.and bin %16, %15 : i1
    cf.cond_br %17, ^bb3, ^bb1(%14, %c0_i6, %false : i64, i6, i1)
  ^bb3:  // pred: ^bb2
    %18 = comb.extract %14 from 32 : (i64) -> i1
    %19 = comb.xor %18, %true : i1
    cf.cond_br %19, ^bb1(%14, %c0_i6, %true : i64, i6, i1), ^bb4
  ^bb4:  // pred: ^bb3
    %20 = comb.extract %3 from 2 : (i192) -> i6
    cf.br ^bb1(%14, %20, %true : i64, i6, i1)
  }
  llhd.drv %_00_, %4#0 after %0 if %4#1 : i6
  %5 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %6 = llhd.prb %_00_ : i6
  llhd.drv %5, %6 after %1 : i6
  %7 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %7, %c0_i186 after %1 : i186
  llhd.drv %clkin_data_0, %clkin_data after %1 : i64
  llhd.drv %in_data_1, %in_data after %1 : i192
  %8 = llhd.prb %out_data : i192
  hw.output %8 : i192
}

// -----// IR Dump Before CSE (cse) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %false = hw.constant false
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %1 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %c0_i192 = hw.constant 0 : i192
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
  %2 = llhd.prb %clkin_data_0 : i64
  %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
  %3 = llhd.prb %in_data_1 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %4:2 = llhd.process -> i6, i1 {
    %9 = llhd.prb %clkin_data_0 : i64
    cf.br ^bb1(%9, %c0_i6, %false : i64, i6, i1)
  ^bb1(%10: i64, %11: i6, %12: i1):  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb4
    %13 = comb.extract %10 from 0 : (i64) -> i1
    llhd.wait yield (%11, %12 : i6, i1), (%2 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %14 = llhd.prb %clkin_data_0 : i64
    %15 = comb.extract %14 from 0 : (i64) -> i1
    %16 = comb.xor bin %13, %true : i1
    %17 = comb.and bin %16, %15 : i1
    cf.cond_br %17, ^bb3, ^bb1(%14, %c0_i6, %false : i64, i6, i1)
  ^bb3:  // pred: ^bb2
    %18 = comb.extract %14 from 32 : (i64) -> i1
    %19 = comb.xor %18, %true : i1
    cf.cond_br %19, ^bb1(%14, %c0_i6, %true : i64, i6, i1), ^bb4
  ^bb4:  // pred: ^bb3
    %20 = comb.extract %3 from 2 : (i192) -> i6
    cf.br ^bb1(%14, %20, %true : i64, i6, i1)
  }
  llhd.drv %_00_, %4#0 after %0 if %4#1 : i6
  %5 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %6 = llhd.prb %_00_ : i6
  llhd.drv %5, %6 after %1 : i6
  %7 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %7, %c0_i186 after %1 : i186
  llhd.drv %clkin_data_0, %clkin_data after %1 : i64
  llhd.drv %in_data_1, %in_data after %1 : i192
  %8 = llhd.prb %out_data : i192
  hw.output %8 : i192
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %false = hw.constant false
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %1 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %c0_i192 = hw.constant 0 : i192
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
  %2 = llhd.prb %clkin_data_0 : i64
  %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
  %3 = llhd.prb %in_data_1 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %4:2 = llhd.process -> i6, i1 {
    %9 = llhd.prb %clkin_data_0 : i64
    cf.br ^bb1(%9, %c0_i6, %false : i64, i6, i1)
  ^bb1(%10: i64, %11: i6, %12: i1):  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb4
    %13 = comb.extract %10 from 0 : (i64) -> i1
    llhd.wait yield (%11, %12 : i6, i1), (%2 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %14 = llhd.prb %clkin_data_0 : i64
    %15 = comb.extract %14 from 0 : (i64) -> i1
    %16 = comb.xor bin %13, %true : i1
    %17 = comb.and bin %16, %15 : i1
    cf.cond_br %17, ^bb3, ^bb1(%14, %c0_i6, %false : i64, i6, i1)
  ^bb3:  // pred: ^bb2
    %18 = comb.extract %14 from 32 : (i64) -> i1
    %19 = comb.xor %18, %true : i1
    cf.cond_br %19, ^bb1(%14, %c0_i6, %true : i64, i6, i1), ^bb4
  ^bb4:  // pred: ^bb3
    %20 = comb.extract %3 from 2 : (i192) -> i6
    cf.br ^bb1(%14, %20, %true : i64, i6, i1)
  }
  llhd.drv %_00_, %4#0 after %0 if %4#1 : i6
  %5 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %6 = llhd.prb %_00_ : i6
  llhd.drv %5, %6 after %1 : i6
  %7 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %7, %c0_i186 after %1 : i186
  llhd.drv %clkin_data_0, %clkin_data after %1 : i64
  llhd.drv %in_data_1, %in_data after %1 : i192
  %8 = llhd.prb %out_data : i192
  hw.output %8 : i192
}

// -----// IR Dump Before RemoveControlFlowPass (llhd-remove-control-flow) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %false = hw.constant false
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %1 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %c0_i192 = hw.constant 0 : i192
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
  %2 = llhd.prb %clkin_data_0 : i64
  %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
  %3 = llhd.prb %in_data_1 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %4:2 = llhd.process -> i6, i1 {
    %9 = llhd.prb %clkin_data_0 : i64
    cf.br ^bb1(%9, %c0_i6, %false : i64, i6, i1)
  ^bb1(%10: i64, %11: i6, %12: i1):  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb4
    %13 = comb.extract %10 from 0 : (i64) -> i1
    llhd.wait yield (%11, %12 : i6, i1), (%2 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %14 = llhd.prb %clkin_data_0 : i64
    %15 = comb.extract %14 from 0 : (i64) -> i1
    %16 = comb.xor bin %13, %true : i1
    %17 = comb.and bin %16, %15 : i1
    cf.cond_br %17, ^bb3, ^bb1(%14, %c0_i6, %false : i64, i6, i1)
  ^bb3:  // pred: ^bb2
    %18 = comb.extract %14 from 32 : (i64) -> i1
    %19 = comb.xor %18, %true : i1
    cf.cond_br %19, ^bb1(%14, %c0_i6, %true : i64, i6, i1), ^bb4
  ^bb4:  // pred: ^bb3
    %20 = comb.extract %3 from 2 : (i192) -> i6
    cf.br ^bb1(%14, %20, %true : i64, i6, i1)
  }
  llhd.drv %_00_, %4#0 after %0 if %4#1 : i6
  %5 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %6 = llhd.prb %_00_ : i6
  llhd.drv %5, %6 after %1 : i6
  %7 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %7, %c0_i186 after %1 : i186
  llhd.drv %clkin_data_0, %clkin_data after %1 : i64
  llhd.drv %in_data_1, %in_data after %1 : i192
  %8 = llhd.prb %out_data : i192
  hw.output %8 : i192
}

// -----// IR Dump Before CSE (cse) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %false = hw.constant false
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %1 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %c0_i192 = hw.constant 0 : i192
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
  %2 = llhd.prb %clkin_data_0 : i64
  %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
  %3 = llhd.prb %in_data_1 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %4:2 = llhd.process -> i6, i1 {
    %9 = llhd.prb %clkin_data_0 : i64
    cf.br ^bb1(%9, %c0_i6, %false : i64, i6, i1)
  ^bb1(%10: i64, %11: i6, %12: i1):  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb4
    %13 = comb.extract %10 from 0 : (i64) -> i1
    llhd.wait yield (%11, %12 : i6, i1), (%2 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %14 = llhd.prb %clkin_data_0 : i64
    %15 = comb.extract %14 from 0 : (i64) -> i1
    %16 = comb.xor bin %13, %true : i1
    %17 = comb.and bin %16, %15 : i1
    cf.cond_br %17, ^bb3, ^bb1(%14, %c0_i6, %false : i64, i6, i1)
  ^bb3:  // pred: ^bb2
    %18 = comb.extract %14 from 32 : (i64) -> i1
    %19 = comb.xor %18, %true : i1
    cf.cond_br %19, ^bb1(%14, %c0_i6, %true : i64, i6, i1), ^bb4
  ^bb4:  // pred: ^bb3
    %20 = comb.extract %3 from 2 : (i192) -> i6
    cf.br ^bb1(%14, %20, %true : i64, i6, i1)
  }
  llhd.drv %_00_, %4#0 after %0 if %4#1 : i6
  %5 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %6 = llhd.prb %_00_ : i6
  llhd.drv %5, %6 after %1 : i6
  %7 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %7, %c0_i186 after %1 : i186
  llhd.drv %clkin_data_0, %clkin_data after %1 : i64
  llhd.drv %in_data_1, %in_data after %1 : i192
  %8 = llhd.prb %out_data : i192
  hw.output %8 : i192
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %false = hw.constant false
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %1 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %c0_i192 = hw.constant 0 : i192
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
  %2 = llhd.prb %clkin_data_0 : i64
  %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
  %3 = llhd.prb %in_data_1 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %4:2 = llhd.process -> i6, i1 {
    %9 = llhd.prb %clkin_data_0 : i64
    cf.br ^bb1(%9, %c0_i6, %false : i64, i6, i1)
  ^bb1(%10: i64, %11: i6, %12: i1):  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb4
    %13 = comb.extract %10 from 0 : (i64) -> i1
    llhd.wait yield (%11, %12 : i6, i1), (%2 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %14 = llhd.prb %clkin_data_0 : i64
    %15 = comb.extract %14 from 0 : (i64) -> i1
    %16 = comb.xor bin %13, %true : i1
    %17 = comb.and bin %16, %15 : i1
    cf.cond_br %17, ^bb3, ^bb1(%14, %c0_i6, %false : i64, i6, i1)
  ^bb3:  // pred: ^bb2
    %18 = comb.extract %14 from 32 : (i64) -> i1
    %19 = comb.xor %18, %true : i1
    cf.cond_br %19, ^bb1(%14, %c0_i6, %true : i64, i6, i1), ^bb4
  ^bb4:  // pred: ^bb3
    %20 = comb.extract %3 from 2 : (i192) -> i6
    cf.br ^bb1(%14, %20, %true : i64, i6, i1)
  }
  llhd.drv %_00_, %4#0 after %0 if %4#1 : i6
  %5 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %6 = llhd.prb %_00_ : i6
  llhd.drv %5, %6 after %1 : i6
  %7 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %7, %c0_i186 after %1 : i186
  llhd.drv %clkin_data_0, %clkin_data after %1 : i64
  llhd.drv %in_data_1, %in_data after %1 : i192
  %8 = llhd.prb %out_data : i192
  hw.output %8 : i192
}

// -----// IR Dump Before MapArithToCombPass (map-arith-to-comb) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %false = hw.constant false
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %1 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %c0_i192 = hw.constant 0 : i192
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
  %2 = llhd.prb %clkin_data_0 : i64
  %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
  %3 = llhd.prb %in_data_1 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %4:2 = llhd.process -> i6, i1 {
    %9 = llhd.prb %clkin_data_0 : i64
    cf.br ^bb1(%9, %c0_i6, %false : i64, i6, i1)
  ^bb1(%10: i64, %11: i6, %12: i1):  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb4
    %13 = comb.extract %10 from 0 : (i64) -> i1
    llhd.wait yield (%11, %12 : i6, i1), (%2 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %14 = llhd.prb %clkin_data_0 : i64
    %15 = comb.extract %14 from 0 : (i64) -> i1
    %16 = comb.xor bin %13, %true : i1
    %17 = comb.and bin %16, %15 : i1
    cf.cond_br %17, ^bb3, ^bb1(%14, %c0_i6, %false : i64, i6, i1)
  ^bb3:  // pred: ^bb2
    %18 = comb.extract %14 from 32 : (i64) -> i1
    %19 = comb.xor %18, %true : i1
    cf.cond_br %19, ^bb1(%14, %c0_i6, %true : i64, i6, i1), ^bb4
  ^bb4:  // pred: ^bb3
    %20 = comb.extract %3 from 2 : (i192) -> i6
    cf.br ^bb1(%14, %20, %true : i64, i6, i1)
  }
  llhd.drv %_00_, %4#0 after %0 if %4#1 : i6
  %5 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %6 = llhd.prb %_00_ : i6
  llhd.drv %5, %6 after %1 : i6
  %7 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %7, %c0_i186 after %1 : i186
  llhd.drv %clkin_data_0, %clkin_data after %1 : i64
  llhd.drv %in_data_1, %in_data after %1 : i192
  %8 = llhd.prb %out_data : i192
  hw.output %8 : i192
}

// -----// IR Dump Before CombineDrivesPass (llhd-combine-drives) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %false = hw.constant false
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %true = hw.constant true
  %c6_i8 = hw.constant 6 : i8
  %1 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %c0_i192 = hw.constant 0 : i192
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
  %2 = llhd.prb %clkin_data_0 : i64
  %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
  %3 = llhd.prb %in_data_1 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %4:2 = llhd.process -> i6, i1 {
    %9 = llhd.prb %clkin_data_0 : i64
    cf.br ^bb1(%9, %c0_i6, %false : i64, i6, i1)
  ^bb1(%10: i64, %11: i6, %12: i1):  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb4
    %13 = comb.extract %10 from 0 : (i64) -> i1
    llhd.wait yield (%11, %12 : i6, i1), (%2 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %14 = llhd.prb %clkin_data_0 : i64
    %15 = comb.extract %14 from 0 : (i64) -> i1
    %16 = comb.xor bin %13, %true : i1
    %17 = comb.and bin %16, %15 : i1
    cf.cond_br %17, ^bb3, ^bb1(%14, %c0_i6, %false : i64, i6, i1)
  ^bb3:  // pred: ^bb2
    %18 = comb.extract %14 from 32 : (i64) -> i1
    %19 = comb.xor %18, %true : i1
    cf.cond_br %19, ^bb1(%14, %c0_i6, %true : i64, i6, i1), ^bb4
  ^bb4:  // pred: ^bb3
    %20 = comb.extract %3 from 2 : (i192) -> i6
    cf.br ^bb1(%14, %20, %true : i64, i6, i1)
  }
  llhd.drv %_00_, %4#0 after %0 if %4#1 : i6
  %5 = llhd.sig.extract %out_data from %c0_i8 : <i192> -> <i6>
  %6 = llhd.prb %_00_ : i6
  llhd.drv %5, %6 after %1 : i6
  %7 = llhd.sig.extract %out_data from %c6_i8 : <i192> -> <i186>
  llhd.drv %7, %c0_i186 after %1 : i186
  llhd.drv %clkin_data_0, %clkin_data after %1 : i64
  llhd.drv %in_data_1, %in_data after %1 : i192
  %8 = llhd.prb %out_data : i192
  hw.output %8 : i192
}

// -----// IR Dump Before Sig2Reg (llhd-sig2reg) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %false = hw.constant false
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %true = hw.constant true
  %1 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i192 = hw.constant 0 : i192
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
  %2 = llhd.prb %clkin_data_0 : i64
  %in_data_1 = llhd.sig name "in_data" %c0_i192 : i192
  %3 = llhd.prb %in_data_1 : i192
  %out_data = llhd.sig %c0_i192 : i192
  %4 = comb.concat %c0_i186, %6 : i186, i6
  llhd.drv %out_data, %4 after %1 : i192
  %5:2 = llhd.process -> i6, i1 {
    %8 = llhd.prb %clkin_data_0 : i64
    cf.br ^bb1(%8, %c0_i6, %false : i64, i6, i1)
  ^bb1(%9: i64, %10: i6, %11: i1):  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb4
    %12 = comb.extract %9 from 0 : (i64) -> i1
    llhd.wait yield (%10, %11 : i6, i1), (%2 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %13 = llhd.prb %clkin_data_0 : i64
    %14 = comb.extract %13 from 0 : (i64) -> i1
    %15 = comb.xor bin %12, %true : i1
    %16 = comb.and bin %15, %14 : i1
    cf.cond_br %16, ^bb3, ^bb1(%13, %c0_i6, %false : i64, i6, i1)
  ^bb3:  // pred: ^bb2
    %17 = comb.extract %13 from 32 : (i64) -> i1
    %18 = comb.xor %17, %true : i1
    cf.cond_br %18, ^bb1(%13, %c0_i6, %true : i64, i6, i1), ^bb4
  ^bb4:  // pred: ^bb3
    %19 = comb.extract %3 from 2 : (i192) -> i6
    cf.br ^bb1(%13, %19, %true : i64, i6, i1)
  }
  llhd.drv %_00_, %5#0 after %0 if %5#1 : i6
  %6 = llhd.prb %_00_ : i6
  llhd.drv %clkin_data_0, %clkin_data after %1 : i64
  llhd.drv %in_data_1, %in_data after %1 : i192
  %7 = llhd.prb %out_data : i192
  hw.output %7 : i192
}

// -----// IR Dump Before CSE (cse) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %false = hw.constant false
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %true = hw.constant true
  %1 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i192 = hw.constant 0 : i192
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
  %2 = llhd.prb %clkin_data_0 : i64
  %c-1_i192 = hw.constant -1 : i192
  %c0_i192_1 = hw.constant 0 : i192
  %c-1_i192_2 = hw.constant -1 : i192
  %c0_i192_3 = hw.constant 0 : i192
  %c0_i192_4 = hw.constant 0 : i192
  %c-1_i192_5 = hw.constant -1 : i192
  %c0_i192_6 = hw.constant 0 : i192
  %c-1_i192_7 = hw.constant -1 : i192
  %c0_i192_8 = hw.constant 0 : i192
  %c0_i192_9 = hw.constant 0 : i192
  %3 = comb.concat %c0_i186, %5 : i186, i6
  %4:2 = llhd.process -> i6, i1 {
    %6 = llhd.prb %clkin_data_0 : i64
    cf.br ^bb1(%6, %c0_i6, %false : i64, i6, i1)
  ^bb1(%7: i64, %8: i6, %9: i1):  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb4
    %10 = comb.extract %7 from 0 : (i64) -> i1
    llhd.wait yield (%8, %9 : i6, i1), (%2 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %11 = llhd.prb %clkin_data_0 : i64
    %12 = comb.extract %11 from 0 : (i64) -> i1
    %13 = comb.xor bin %10, %true : i1
    %14 = comb.and bin %13, %12 : i1
    cf.cond_br %14, ^bb3, ^bb1(%11, %c0_i6, %false : i64, i6, i1)
  ^bb3:  // pred: ^bb2
    %15 = comb.extract %11 from 32 : (i64) -> i1
    %16 = comb.xor %15, %true : i1
    cf.cond_br %16, ^bb1(%11, %c0_i6, %true : i64, i6, i1), ^bb4
  ^bb4:  // pred: ^bb3
    %17 = comb.extract %in_data from 2 : (i192) -> i6
    cf.br ^bb1(%11, %17, %true : i64, i6, i1)
  }
  llhd.drv %_00_, %4#0 after %0 if %4#1 : i6
  %5 = llhd.prb %_00_ : i6
  llhd.drv %clkin_data_0, %clkin_data after %1 : i64
  hw.output %3 : i192
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %false = hw.constant false
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %true = hw.constant true
  %1 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
  %2 = llhd.prb %clkin_data_0 : i64
  %3 = comb.concat %c0_i186, %5 : i186, i6
  %4:2 = llhd.process -> i6, i1 {
    %6 = llhd.prb %clkin_data_0 : i64
    cf.br ^bb1(%6, %c0_i6, %false : i64, i6, i1)
  ^bb1(%7: i64, %8: i6, %9: i1):  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb4
    %10 = comb.extract %7 from 0 : (i64) -> i1
    llhd.wait yield (%8, %9 : i6, i1), (%2 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %11 = llhd.prb %clkin_data_0 : i64
    %12 = comb.extract %11 from 0 : (i64) -> i1
    %13 = comb.xor bin %10, %true : i1
    %14 = comb.and bin %13, %12 : i1
    cf.cond_br %14, ^bb3, ^bb1(%11, %c0_i6, %false : i64, i6, i1)
  ^bb3:  // pred: ^bb2
    %15 = comb.extract %11 from 32 : (i64) -> i1
    %16 = comb.xor %15, %true : i1
    cf.cond_br %16, ^bb1(%11, %c0_i6, %true : i64, i6, i1), ^bb4
  ^bb4:  // pred: ^bb3
    %17 = comb.extract %in_data from 2 : (i192) -> i6
    cf.br ^bb1(%11, %17, %true : i64, i6, i1)
  }
  llhd.drv %_00_, %4#0 after %0 if %4#1 : i6
  %5 = llhd.prb %_00_ : i6
  llhd.drv %clkin_data_0, %clkin_data after %1 : i64
  hw.output %3 : i192
}

// -----// IR Dump Before RegOfVecToMem (seq-reg-of-vec-to-mem) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %false = hw.constant false
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %true = hw.constant true
  %1 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
  %2 = llhd.prb %clkin_data_0 : i64
  %3 = comb.concat %c0_i186, %5 : i186, i6
  %4:2 = llhd.process -> i6, i1 {
    %6 = llhd.prb %clkin_data_0 : i64
    cf.br ^bb1(%6, %c0_i6, %false : i64, i6, i1)
  ^bb1(%7: i64, %8: i6, %9: i1):  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb4
    %10 = comb.extract %7 from 0 : (i64) -> i1
    llhd.wait yield (%8, %9 : i6, i1), (%2 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %11 = llhd.prb %clkin_data_0 : i64
    %12 = comb.extract %11 from 0 : (i64) -> i1
    %13 = comb.xor bin %10, %true : i1
    %14 = comb.and bin %13, %12 : i1
    cf.cond_br %14, ^bb3, ^bb1(%11, %c0_i6, %false : i64, i6, i1)
  ^bb3:  // pred: ^bb2
    %15 = comb.extract %11 from 32 : (i64) -> i1
    %16 = comb.xor %15, %true : i1
    cf.cond_br %16, ^bb1(%11, %c0_i6, %true : i64, i6, i1), ^bb4
  ^bb4:  // pred: ^bb3
    %17 = comb.extract %in_data from 2 : (i192) -> i6
    cf.br ^bb1(%11, %17, %true : i64, i6, i1)
  }
  llhd.drv %_00_, %4#0 after %0 if %4#1 : i6
  %5 = llhd.prb %_00_ : i6
  llhd.drv %clkin_data_0, %clkin_data after %1 : i64
  hw.output %3 : i192
}

// -----// IR Dump Before CSE (cse) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %false = hw.constant false
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %true = hw.constant true
  %1 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
  %2 = llhd.prb %clkin_data_0 : i64
  %3 = comb.concat %c0_i186, %5 : i186, i6
  %4:2 = llhd.process -> i6, i1 {
    %6 = llhd.prb %clkin_data_0 : i64
    cf.br ^bb1(%6, %c0_i6, %false : i64, i6, i1)
  ^bb1(%7: i64, %8: i6, %9: i1):  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb4
    %10 = comb.extract %7 from 0 : (i64) -> i1
    llhd.wait yield (%8, %9 : i6, i1), (%2 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %11 = llhd.prb %clkin_data_0 : i64
    %12 = comb.extract %11 from 0 : (i64) -> i1
    %13 = comb.xor bin %10, %true : i1
    %14 = comb.and bin %13, %12 : i1
    cf.cond_br %14, ^bb3, ^bb1(%11, %c0_i6, %false : i64, i6, i1)
  ^bb3:  // pred: ^bb2
    %15 = comb.extract %11 from 32 : (i64) -> i1
    %16 = comb.xor %15, %true : i1
    cf.cond_br %16, ^bb1(%11, %c0_i6, %true : i64, i6, i1), ^bb4
  ^bb4:  // pred: ^bb3
    %17 = comb.extract %in_data from 2 : (i192) -> i6
    cf.br ^bb1(%11, %17, %true : i64, i6, i1)
  }
  llhd.drv %_00_, %4#0 after %0 if %4#1 : i6
  %5 = llhd.prb %_00_ : i6
  llhd.drv %clkin_data_0, %clkin_data after %1 : i64
  hw.output %3 : i192
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
  %false = hw.constant false
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %true = hw.constant true
  %1 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i64 = hw.constant 0 : i64
  %c0_i186 = hw.constant 0 : i186
  %c0_i6 = hw.constant 0 : i6
  %_00_ = llhd.sig %c0_i6 : i6
  %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
  %2 = llhd.prb %clkin_data_0 : i64
  %3 = comb.concat %c0_i186, %5 : i186, i6
  %4:2 = llhd.process -> i6, i1 {
    %6 = llhd.prb %clkin_data_0 : i64
    cf.br ^bb1(%6, %c0_i6, %false : i64, i6, i1)
  ^bb1(%7: i64, %8: i6, %9: i1):  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb4
    %10 = comb.extract %7 from 0 : (i64) -> i1
    llhd.wait yield (%8, %9 : i6, i1), (%2 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %11 = llhd.prb %clkin_data_0 : i64
    %12 = comb.extract %11 from 0 : (i64) -> i1
    %13 = comb.xor bin %10, %true : i1
    %14 = comb.and bin %13, %12 : i1
    cf.cond_br %14, ^bb3, ^bb1(%11, %c0_i6, %false : i64, i6, i1)
  ^bb3:  // pred: ^bb2
    %15 = comb.extract %11 from 32 : (i64) -> i1
    %16 = comb.xor %15, %true : i1
    cf.cond_br %16, ^bb1(%11, %c0_i6, %true : i64, i6, i1), ^bb4
  ^bb4:  // pred: ^bb3
    %17 = comb.extract %in_data from 2 : (i192) -> i6
    cf.br ^bb1(%11, %17, %true : i64, i6, i1)
  }
  llhd.drv %_00_, %4#0 after %0 if %4#1 : i6
  %5 = llhd.prb %_00_ : i6
  llhd.drv %clkin_data_0, %clkin_data after %1 : i64
  hw.output %3 : i192
}

module {
  hw.module @top_arc(in %clkin_data : i64, in %in_data : i192, out out_data : i192) {
    %false = hw.constant false
    %0 = llhd.constant_time <0ns, 1d, 0e>
    %true = hw.constant true
    %1 = llhd.constant_time <0ns, 0d, 1e>
    %c0_i64 = hw.constant 0 : i64
    %c0_i186 = hw.constant 0 : i186
    %c0_i6 = hw.constant 0 : i6
    %_00_ = llhd.sig %c0_i6 : i6
    %clkin_data_0 = llhd.sig name "clkin_data" %c0_i64 : i64
    %2 = llhd.prb %clkin_data_0 : i64
    %3 = comb.concat %c0_i186, %5 : i186, i6
    %4:2 = llhd.process -> i6, i1 {
      %6 = llhd.prb %clkin_data_0 : i64
      cf.br ^bb1(%6, %c0_i6, %false : i64, i6, i1)
    ^bb1(%7: i64, %8: i6, %9: i1):  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb4
      %10 = comb.extract %7 from 0 : (i64) -> i1
      llhd.wait yield (%8, %9 : i6, i1), (%2 : i64), ^bb2
    ^bb2:  // pred: ^bb1
      %11 = llhd.prb %clkin_data_0 : i64
      %12 = comb.extract %11 from 0 : (i64) -> i1
      %13 = comb.xor bin %10, %true : i1
      %14 = comb.and bin %13, %12 : i1
      cf.cond_br %14, ^bb3, ^bb1(%11, %c0_i6, %false : i64, i6, i1)
    ^bb3:  // pred: ^bb2
      %15 = comb.extract %11 from 32 : (i64) -> i1
      %16 = comb.xor %15, %true : i1
      cf.cond_br %16, ^bb1(%11, %c0_i6, %true : i64, i6, i1), ^bb4
    ^bb4:  // pred: ^bb3
      %17 = comb.extract %in_data from 2 : (i192) -> i6
      cf.br ^bb1(%11, %17, %true : i64, i6, i1)
    }
    llhd.drv %_00_, %4#0 after %0 if %4#1 : i6
    %5 = llhd.prb %_00_ : i6
    llhd.drv %clkin_data_0, %clkin_data after %1 : i64
    hw.output %3 : i192
  }
}
