// RUN: %cladclang %s -I%S/../../include -oTapeMemory.out 2>&1
// RUN: ./TapeMemory.out
// XFAIL: valgrind

#include "clad/Differentiator/Differentiator.h"

#include <thread>
#include <vector>

struct A {
  bool operator!=(const A& other){ return false; }
};

// Minimum viable code to test for tape pushes/pops
template <typename T> void func(T x, int n) {
  clad::tape<T> t = {};
  for (int i = 0; i < n; i++) {
    clad::push<T>(t, x);
  }

  for(auto p = t.begin(); p!=t.end(); ++p)
    if (*p!=x)
      printf("error: tape iterator is invalid\n");

  for (int i = 0; i < n; i++) {
    T seen = clad::pop<T>(t);
    if (seen != x)
      printf("error: tape is invalid!\n");
  }
}

template <typename T>
void concurrent_push_test(T x, int n_threads, int pushes_per_thread) {
  // Use thread_local clad::tape<T> t = {}; and clad::push<T>(t, x); for local thread storage
  clad::tape<T, 64, 1024, true> t = {};
  std::vector<std::thread> threads;

  for (int i = 0; i < n_threads; ++i) {
    threads.emplace_back([&]() {
      for (int j = 0; j < pushes_per_thread; ++j) {
        clad::push<T>(t, x);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  size_t expected = n_threads * pushes_per_thread;
  size_t actual = t.size();
  if (expected != actual) {
    printf("error: expected size %zu, actual size %zu\n", expected, actual);
  }
}

int main() {

  int block = 32, n = 5;
  int dummy = 0;
  for (int i = 0; i < n; i++, block *= 2) {
    // Scalar types
    func<bool>(1, block);
    func<char>(1, block);
    func<int>(1, block);
    func<float>(1, block);
    func<double>(1, block);
    func<long>(1, block);
    func<long double>(1, block);
    // Const/Volatile types
    func<const int>(static_cast<const int>(dummy), block);
    func<volatile float>(static_cast<volatile float>(dummy), block);
    func<const volatile double>(static_cast<const volatile double>(dummy),
                                block);
    // custom type
    func<A>(A(), block);
  }

  for (int i = 0; i < 1000; ++i) {
    concurrent_push_test<int>(1, 8, 1000);
  }
}
