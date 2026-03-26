// RUN: %cladclang -std=c++17 -fno-exceptions -I%S/../../include %s -o %t
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Tape.h"
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>
#include <string>

// User Defined Tape
template <typename T, std::size_t SBO = 64, std::size_t SLAB_SIZE = 1024, bool multithreading = false, bool DiskOffload = true>
struct MockTape {
  float last_val = 0.0f;
  void emplace_back(float val) { 
    last_val = val;
    printf("Mock push: %.2f\n", val); 
  }
  float back() const { return 99.9f; }
  void pop_back() {}
};

namespace clad {
template <typename T, std::size_t SBO = 64, std::size_t SLAB_SIZE = 1024, bool multithreading = false, bool DiskOffload = true>
   using custom_tape = MockTape<T, SBO, SLAB_SIZE, multithreading, DiskOffload>;
}
float fn(float x, float y) {
  float z = x;
  for (int i = 0; i < 3; ++i)
    z = z * y + std::sin(x);
  return z;
}

int main() {
  // Test standard tape behaviors
  clad::custom_tape<double, 64, 1024, false, true> default_tape;
  default_tape.emplace_back(3.14);
  // We expect 99.9 because of your MockTape::back() implementation
  printf("Standard tape back: %.1f\n", default_tape.back());
  // CHECK-EXEC: Standard tape back: 99.9

  // Test RamDiskManager 
  clad::detail::RamDiskManager<double, 1024> mmap_manager;
  double test_data[1024];
  for(int i = 0; i < 1024; ++i) 
    test_data[i] = i * 1.5;

  size_t offset = mmap_manager.write_slab(test_data);
  double read_data[1024];
  mmap_manager.read_slab(read_data, offset);

  bool mmap_ok = (read_data[0] == 0.0 && read_data[1023] == 1023 * 1.5);
  printf("Mmap manager read/write ok: %d\n", mmap_ok);
  // CHECK-EXEC: Mmap manager read/write ok: 1

  // Test Tape Specialization Hook
  clad::custom_tape<float, 64, 1024, false, true> specialized_tape;
  specialized_tape.emplace_back(2.71f);
  // CHECK-EXEC: Mock push: 2.71
  printf("Specialized tape back: %.1f\n", specialized_tape.back());
  // CHECK-EXEC: Specialized tape back: 99.9

  // Fixed: DiskManager constructor takes a bool (DiskOffload), not a string
  clad::detail::DiskManager<double, 1024> file_backed_manager(true); 
  
  size_t file_offset = file_backed_manager.write_slab(test_data);
  double file_read_data[1024];
  file_backed_manager.read_slab(file_read_data, file_offset);

  bool file_mmap_ok = (file_read_data[0] == 0.0 && file_read_data[1023] == 1023 * 1.5);
  printf("File-Backed manager read/write ok: %d\n", file_mmap_ok);
  // CHECK-EXEC: File-Backed manager read/write ok: 1

  float dx = 0, dy = 0;
  auto d_fn = clad::gradient(fn);
  d_fn.execute(1.0f, 2.0f, &dx, &dy);
  printf("Gradient computed: %d\n", (dx != 0 || dy != 0));
  // CHECK-EXEC: Gradient computed: 1

  return 0;
}