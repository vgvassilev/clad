// Recover light emission with the complete standalone demo. Keep generated
// images and trajectory data in this test's temporary directory.
// RUN: %cladclang %S/../../demos/ComputerGraphics/smallpt/SmallPTDiff.cpp -I%S/../../include -o%t
// RUN: mkdir -p %t.dir
// RUN: cd %t.dir && %t | %filecheck_exec %s
// XFAIL: valgrind
// UNSUPPORTED: target={{i586.*}}
// CHECK-EXEC: Target light_e=1.0000
// CHECK-EXEC: Start  light_e=0.3500
// CHECK-EXEC: Final light_e=
// CHECK-EXEC: SMALLPT_DIFF_INVERSE_PASS=1
