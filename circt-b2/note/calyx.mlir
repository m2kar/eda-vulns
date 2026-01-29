module attributes {calyx.entrypoint = "f"} {
  calyx.component @f(%clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
    %c0_i32 = hw.constant 0 : i32
    %c1_i32 = hw.constant 1 : i32
    %c10_i32 = hw.constant 10 : i32
    %true = hw.constant true
    %std_add_1.left, %std_add_1.right, %std_add_1.out = calyx.std_add @std_add_1 : i32, i32, i32
    %std_add_0.left, %std_add_0.right, %std_add_0.out = calyx.std_add @std_add_0 : i32, i32, i32
    %std_slt_0.left, %std_slt_0.right, %std_slt_0.out = calyx.std_slt @std_slt_0 : i32, i32, i1
    %while_0_arg1_reg.in, %while_0_arg1_reg.write_en, %while_0_arg1_reg.clk, %while_0_arg1_reg.reset, %while_0_arg1_reg.out, %while_0_arg1_reg.done = calyx.register @while_0_arg1_reg : i32, i1, i1, i1, i32, i1
    %while_0_arg0_reg.in, %while_0_arg0_reg.write_en, %while_0_arg0_reg.clk, %while_0_arg0_reg.reset, %while_0_arg0_reg.out, %while_0_arg0_reg.done = calyx.register @while_0_arg0_reg : i32, i1, i1, i1, i32, i1
    %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
    calyx.wires {
      calyx.assign %out0 = %ret_arg0_reg.out : i32
      calyx.group @assign_while_0_init_0 {
        calyx.assign %while_0_arg0_reg.in = %c0_i32 : i32
        calyx.assign %while_0_arg0_reg.write_en = %true : i1
        calyx.group_done %while_0_arg0_reg.done : i1
      }
      calyx.group @assign_while_0_init_1 {
        calyx.assign %while_0_arg1_reg.in = %c0_i32 : i32
        calyx.assign %while_0_arg1_reg.write_en = %true : i1
        calyx.group_done %while_0_arg1_reg.done : i1
      }
      calyx.comb_group @bb0_0 {
        calyx.assign %std_slt_0.left = %while_0_arg0_reg.out : i32
        calyx.assign %std_slt_0.right = %c10_i32 : i32
      }
      calyx.group @assign_while_0_latch {
        calyx.assign %while_0_arg0_reg.in = %std_add_0.out : i32
        calyx.assign %while_0_arg0_reg.write_en = %true : i1
        calyx.assign %while_0_arg1_reg.in = %std_add_1.out : i32
        calyx.assign %while_0_arg1_reg.write_en = %true : i1
        calyx.assign %std_add_0.left = %while_0_arg0_reg.out : i32
        calyx.assign %std_add_0.right = %c1_i32 : i32
        calyx.assign %std_add_1.left = %while_0_arg1_reg.out : i32
        calyx.assign %std_add_1.right = %c1_i32 : i32
        %0 = comb.and %while_0_arg1_reg.done, %while_0_arg0_reg.done : i1
        calyx.group_done %0 ? %true : i1
      }
      calyx.group @ret_assign_0 {
        calyx.assign %ret_arg0_reg.in = %while_0_arg1_reg.out : i32
        calyx.assign %ret_arg0_reg.write_en = %true : i1
        calyx.group_done %ret_arg0_reg.done : i1
      }
    }
    calyx.control {
      calyx.seq {
        calyx.seq {
          calyx.par {
            calyx.enable @assign_while_0_init_0
            calyx.enable @assign_while_0_init_1
          }
          calyx.while %std_slt_0.out with @bb0_0 {
            calyx.seq {
              calyx.enable @assign_while_0_latch
            }
          }
          calyx.enable @ret_assign_0
        }
      }
    }
  } {toplevel}
}

