// RUN: %cladclang %s -I%S/../../include -oRecordRange.out
// RUN: ./RecordRange.out | %filecheck_exec %s

// clad::record_range / peek_range / drop_range are how a call site puts back a
// range a callee overwrote, in place of a restore_tracker's address-keyed log.
// peek_range must be non-destructive and must be replayable twice, because a
// call's reverse sweep restores once before the pullback and again after --
// the pullback's own replay mutates what the first restore put back.
//
// The sizes below straddle the tape's inline capacity and its slab size, which
// is where peek_back's index arithmetic has to walk back through slabs rather
// than index the tail directly.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>
#include <vector>

int main() {
  const std::size_t totals[] = {5, 64, 65, 1000, 1024, 1089, 5000};
  const std::size_t runs[] = {1, 3, 64, 100, 1500};

  for (std::size_t total : totals) {
    for (std::size_t n : runs) {
      if (n > total)
        continue;
      clad::tape<double> t = {};
      for (std::size_t i = 0; i < total; i++)
        clad::push(t, (double)i);

      std::vector<double> got(n, -1);
      clad::peek_range(t, got.data(), n);
      for (std::size_t i = 0; i < n; i++)
        if (got[i] != (double)(total - n + i)) {
          printf("FAIL value total=%zu n=%zu\n", total, n);
          return 1;
        }
      if (t.size() != total) {
        printf("FAIL peek consumed total=%zu n=%zu\n", total, n);
        return 1;
      }

      // Replaying the same run again must give the same values.
      std::vector<double> again(n, -1);
      clad::peek_range(t, again.data(), n);
      for (std::size_t i = 0; i < n; i++)
        if (again[i] != got[i]) {
          printf("FAIL second peek total=%zu n=%zu\n", total, n);
          return 1;
        }

      clad::drop_range(t, n);
      if (t.size() != total - n) {
        printf("FAIL drop total=%zu n=%zu\n", total, n);
        return 1;
      }
      if (total - n > 0 && clad::back(t) != (double)(total - n - 1)) {
        printf("FAIL back after drop total=%zu n=%zu\n", total, n);
        return 1;
      }
    }
  }

  // record_range is the producer the three above consume.
  double src[4] = {1, 2, 3, 4};
  clad::tape<double> t = {};
  clad::record_range(t, src, 4);
  double dst[4] = {0, 0, 0, 0};
  clad::peek_range(t, dst, 4);
  printf("%.0f %.0f %.0f %.0f\n", dst[0], dst[1], dst[2], dst[3]);
  // CHECK-EXEC: 1 2 3 4

  printf("record_range OK\n");
  // CHECK-EXEC: record_range OK
  return 0;
}
