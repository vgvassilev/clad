// A type clad cannot take apart, measured instead of derived. It needs
// numerical differentiation, which %cladclang compiles out.
//
// RUN: %cladnumdiffclang %S/../../demos/CustomTypeNumDiff.cpp -I%S/../../include -o%t
// RUN: %t | %filecheck_exec %s

// CHECK-EXEC: Result of df/dx is = 0.07
// CHECK-EXEC: Result of df/dx is = 0.003
