#ifndef CLAD_DIFFERENTIATOR_RESTORETRACKER_H
#define CLAD_DIFFERENTIATOR_RESTORETRACKER_H

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <map>
#include <thread>
#include <utility>
#include <vector>

namespace clad {

/// This class is used for bitwise storing/restoring variables.
/// It is passed to reverse_forw to store the state of the program before the
/// function call, for example,
/// f_reverse_forw(..., _tracker0);
/// ...
/// _tracker0.restore();
/// f_pullback(...);
/// We use it when we have to pass information between nested calls and
/// clad::tape is not viable.
class restore_tracker {
  // m_data consists of pairs of memory addresses and bitwise values
  using RawMemory = std::vector<uint8_t>;
  using Address = char*;
  std::map<const Address, RawMemory> m_data;

public:
  // Store the value and the address of `val`.
  template <typename T> void store(const T& val) {
    // If a variable is stored multiple times, we should only take the first
    // value into consideration:
    // _tracker.store(x); // stored
    // ...
    // _tracker.store(x); // ignored
    if (m_data.find((char*)&val) != m_data.end())
      return;
    std::vector<uint8_t> buffer(sizeof(T));
    std::memcpy(buffer.data(), &val, sizeof(T));
    m_data.emplace((char*)&val, std::move(buffer));
  }
  // Set all stored addresses to the corresponsing values bitwise.
  void restore() {
    size_t total_items = m_data.size();
    if (total_items == 0)
      return;
    size_t avail_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;
    workers.reserve(avail_threads);
    size_t task_per_worker = total_items / avail_threads;
    size_t left_task = total_items % avail_threads;
    auto block_start = m_data.begin();
    for (size_t i = 0; i < avail_threads; i++) {
      size_t temp = task_per_worker + (i < left_task ? 1 : 0);
      auto block_end = block_start;
      std::advance(block_end, temp);
      workers.emplace_back([block_start, block_end]() {
        for (auto it = block_start; it != block_end; it++)
          std::memcpy(it->first, it->second.data(), it->second.size());
      });
      block_start = block_end;
    }
    for (auto& it : workers)
      if (it.joinable())
        it.join();

    m_data.clear();
  }
};
} // namespace clad

#endif // CLAD_DIFFERENTIATOR_RESTORETRACKER_H
