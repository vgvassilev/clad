// RUN: %cladclang %s -I%S/../../include -o %t
// RUN: %t | %filecheck_exec %s
// RUN: %cladclang -std=c++14 %s -I%S/../../include -o %t14
// RUN: %t14 | %filecheck_exec %s
// RUN: %cladclang -std=c++20 %s -I%S/../../include -o %t20
// RUN: %t20 | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <array>
#include <cstddef>
#include <cstdio>
#include <deque>
#include <forward_list>
#include <initializer_list>
#include <list>
#include <string>
#include <valarray>
#include <vector>

namespace zero_like_test {
struct Counted {
  double value = 7.;
  static long copies;

  Counted() = default;
  Counted(const Counted& other) : value(other.value) { ++copies; }
  Counted& operator=(const Counted&) = default;
};

long Counted::copies = 0;

void zero_init(Counted& value) { value.value = 0.; }
} // namespace zero_like_test

using zero_like_test::Counted;

// Models containers such as unordered_map, where T(size_type) configures
// buckets rather than constructing size() elements.
class BucketCountRange {
public:
  using size_type = std::size_t;

  BucketCountRange(std::initializer_list<double> values) : storage(values) {}
  explicit BucketCountRange(size_type) {}

  std::vector<double>::iterator begin() { return storage.begin(); }
  std::vector<double>::iterator end() { return storage.end(); }
  std::vector<double>::const_iterator begin() const { return storage.begin(); }
  std::vector<double>::const_iterator end() const { return storage.end(); }
  size_type size() const { return storage.size(); }

private:
  std::vector<double> storage;
};

template <class Range> bool is_zero(const Range& range) {
  for (const auto& value : range)
    if (value != 0)
      return false;
  return true;
}

int main() {
  std::array<double, 2> array{{1., 2.}};
  std::deque<double> deque{1., 2.};
  std::forward_list<double> forwardList{1., 2.};
  std::list<double> list{1., 2.};
  std::vector<double> vector{1., 2.};
  std::valarray<double> valarray{1., 2.};
  std::string string{'a', 'b'};

  std::printf("%d %d %d %d %d %d %d\n",
              is_zero(clad::zero_like(array)),
              is_zero(clad::zero_like(deque)),
              is_zero(clad::zero_like(forwardList)),
              is_zero(clad::zero_like(list)),
              is_zero(clad::zero_like(vector)),
              is_zero(clad::zero_like(valarray)),
              is_zero(clad::zero_like(string)));

  std::vector<std::vector<Counted>> ragged(3);
  ragged[0].resize(2);
  ragged[1].resize(1);
  ragged[2].resize(3);
  for (auto& row : ragged)
    for (auto& value : row)
      value.value = 3.14;

  Counted::copies = 0;
  auto dRagged = clad::zero_like(ragged);
  bool sameShapeAndZero = dRagged.size() == ragged.size();
  for (std::size_t i = 0; i < ragged.size(); ++i) {
    sameShapeAndZero &= dRagged[i].size() == ragged[i].size();
    for (const auto& value : dRagged[i])
      sameShapeAndZero &= value.value == 0.;
  }
  std::printf("%d %ld\n", sameShapeAndZero, Counted::copies);

  BucketCountRange bucketRange{1., 2.};
  auto dBucketRange = clad::zero_like(bucketRange);
  std::printf("%d\n", dBucketRange.size() == bucketRange.size() &&
                              is_zero(dBucketRange));
}

// CHECK-EXEC: 1 1 1 1 1 1 1
// CHECK-EXEC-NEXT: 1 0
// CHECK-EXEC-NEXT: 1
