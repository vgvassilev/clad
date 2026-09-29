// A network with one hidden layer learning exclusive or from nothing but the
// gradient of its loss. The test asks whether it learned: the four answers
// have to come out on the right side of zero, which they do not if the
// gradient is wrong.
//
// RUN: %cladclang %S/../../demos/XorNetwork.cpp -I%S/../../include -o%t 2>&1 \
// RUN:     | %filecheck %s
// RUN: %t | %filecheck_exec %s

// CHECK: void loss_grad(const double w[9], double *_d_w) {

// CHECK-EXEC: step 4000: loss = 0.0{{[0-9]+}}
// CHECK-EXEC: 0 XOR 0 -> -0.9{{[0-9]+}} (want -1)
// CHECK-EXEC: 0 XOR 1 -> +0.9{{[0-9]+}} (want +1)
// CHECK-EXEC: 1 XOR 0 -> +0.9{{[0-9]+}} (want +1)
// CHECK-EXEC: 1 XOR 1 -> -0.9{{[0-9]+}} (want -1)
