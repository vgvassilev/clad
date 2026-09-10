// RUN: %cladclang %s -I%S/../../include -Xclang -verify -o %t
// RUN: %t
// expected-no-diagnostics

#include "clad/Differentiator/Differentiator.h"
#include <cassert>
#include <memory>

struct Tracked {
  static int live;
  Tracked() { ++live; }
  ~Tracked() { --live; }
};

int Tracked::live = 0;

template <bool ThreadSafe> void test_tape() {
  const int values[2][1][3] = {{{1, 2, 3}}, {{4, 5, 6}}};
  clad::tape<int[2][1][3], 2, 2, ThreadSafe> tape;
  // Cross both the inline-storage and slab boundaries.
  for (int i = 0; i < 7; ++i) {
    clad::push(tape, values);
    assert(tape.back()[1][0][2] == 6);
    tape.back()[0][0][0] = i;
  }
  for (int i = 6; i >= 0; --i) {
    assert(tape.back()[0][0][0] == i);
    clad::pop(tape);
  }
  assert(tape.size() == 0);

  {
    clad::tape<Tracked[2][2], 2, 2, ThreadSafe> records;
    for (int i = 0; i < 7; ++i)
      records.emplace_back();
    assert(Tracked::live == 28);
    clad::pop(records);
    assert(Tracked::live == 24);
  }
  assert(Tracked::live == 0);
}

int main() {
  int vector[2] = {1, 2};
  int vector_copy[2] = {};
  clad::move<int, 2>(vector, vector_copy);
  assert(vector_copy[0] == 1 && vector_copy[1] == 2);

  const int input[2][3] = {{1, 2, 3}, {4, 5, 6}};
  int output[2][3] = {};
  clad::move(input, output);
  assert(output[0][2] == 3 && output[1][2] == 6);

  // A short initializer resets the remaining subarrays as well.
  clad::move({{7, 8}}, output);
  assert(output[0][0] == 7 && output[0][1] == 8 && output[0][2] == 0);
  assert(output[1][0] == 0 && output[1][1] == 0 && output[1][2] == 0);

  std::unique_ptr<int> pointers[1][2];
  pointers[0][0] = std::make_unique<int>(11);
  pointers[0][1] = std::make_unique<int>(13);
  std::unique_ptr<int> moved[1][2];
  clad::move(pointers, moved);
  assert(!pointers[0][0] && !pointers[0][1]);
  assert(*moved[0][0] == 11 && *moved[0][1] == 13);

  test_tape<false>();
  test_tape<true>();
}
