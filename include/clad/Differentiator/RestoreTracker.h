#ifndef CLAD_DIFFERENTIATOR_RESTORETRACKER_H
#define CLAD_DIFFERENTIATOR_RESTORETRACKER_H

#include <cassert>
#include <cstdint>
#include <cstring>
#include <unordered_set>
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
  std::unordered_set<void*> m_Seen;
  std::vector<uint8_t> m_Buffer;

public:
  restore_tracker() {
    // Prior reservation of some buffer for small and medium workloads will save
    // us from massive reallocation overhead
    m_Buffer.reserve(4096);
  }
  // buffer stores three things: address|size|raw bytes of val
  template <typename T> void store(const T& val) {
    void* addr = (void*)&val;
    // If a variable is stored multiple times, we should only take the first
    // value into consideration:
    // _tracker.store(x); // stored
    // ...
    // _tracker.store(x); // ignored
    if (!m_Seen.insert(addr).second)
      return;

    size_t size = sizeof(T);
    size_t current_size = m_Buffer.size();

    m_Buffer.resize(current_size + sizeof(void*) + sizeof(size_t) + size);

    uint8_t* ptr = m_Buffer.data() + current_size;
    std::memcpy(ptr, &addr, sizeof(void*));
    ptr += sizeof(void*);

    std::memcpy(ptr, &size, sizeof(size_t));
    ptr += sizeof(size_t);

    std::memcpy(ptr, addr, size);
  }

  template <typename = void> void restore() {
    if (m_Buffer.empty())
      return;

    uint8_t* ptr = m_Buffer.data();
    uint8_t* end = ptr + m_Buffer.size();

    while (ptr < end) {
      void* addr = nullptr;
      size_t size = 0;

      std::memcpy(&addr, ptr, sizeof(void*));
      ptr += sizeof(void*);

      std::memcpy(&size, ptr, sizeof(size_t));
      ptr += sizeof(size_t);

      std::memcpy(addr, ptr, size);
      ptr += size;
    }

    m_Buffer.clear();
    m_Seen.clear();
  }
};
} // namespace clad

#endif // CLAD_DIFFERENTIATOR_RESTORETRACKER_H