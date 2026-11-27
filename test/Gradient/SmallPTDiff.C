// RUN: %cladclang %s -I%S/../../include -I%S/../../demos/ComputerGraphics/smallpt -oSmallPTDiff.out
// RUN: ./SmallPTDiff.out | %filecheck_exec %s
// XFAIL: valgrind
// FD check is not reliable on 32-bit x86.
// UNSUPPORTED: target={{i586.*}}

#include "SmallPTDiffValidate.h"

int main() {
  int fail = 0;
  fail |= smallpt_run_camera_grad_check();
  fail |= smallpt_run_patch_grad_check(2, 2, 1);
  fail |= smallpt_run_light_y_grad_check();
  fail |= smallpt_run_mirror_patch_grad_check();
  fail |= smallpt_run_light_e_grad_check();
  fail |= smallpt_run_light_e_gd_smoke();
  printf("SMALLPT_DIFF_PASS=%d\n", fail ? 0 : 1);
  return fail;
}

// CHECK-EXEC: SMALLPT_GRAD_FD_PASS=1
// CHECK-EXEC: SMALLPT_PATCH_GRAD_FD_PASS=1
// CHECK-EXEC: SMALLPT_LIGHT_Y_GRAD_FD_PASS=1
// CHECK-EXEC: PATCH_MIRROR_RADIANCE=
// CHECK-EXEC-NOT: PATCH_MIRROR_RADIANCE_FAIL=1
// CHECK-EXEC: SMALLPT_MIRROR_PATCH_GRAD_FD_PASS=1
// CHECK-EXEC: SMALLPT_LIGHT_E_GRAD_FD_PASS=1
// CHECK-EXEC: SMALLPT_LIGHT_E_GD_SMOKE_PASS=1
// CHECK-EXEC: SMALLPT_DIFF_PASS=1
