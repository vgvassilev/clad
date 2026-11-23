// In a CUDA build restore_tracker keeps what it stores in a fixed buffer,
// on the host as well as on the device. A value that does not fit used to
// be dropped without a word, and so was never restored. Now it is reported.
// The host half is what runs here; a trap on the device would take the CUDA
// context down with it.

// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -o RestoreTrackerOverflow.out %s
//
// RUN: %if cuda-runtime %{ not ./RestoreTrackerOverflow.out | %filecheck_exec %s %}
//
// REQUIRES: cuda-compile
#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

int main() {
  clad::restore_tracker tracker;

  double fits = 1;
  tracker.store(fits);
  fits = 2;
  tracker.restore();
  printf("%g\n", fits); // CHECK-EXEC: 1

  // One byte more than the whole buffer holds.
  static char too_big[Max_Bytes + 1];
  tracker.store(too_big);
  // CHECK-EXEC: clad: restore_tracker is full; compile with a larger Max_Records or Max_Bytes
  printf("stored\n"); // CHECK-EXEC-NOT: stored
  return 0;
}
