#ifndef CLAD_DIFFERENTIATOR_RESTORETRACKER_H
#define CLAD_DIFFERENTIATOR_RESTORETRACKER_H

// CUDA_HOST_DEVICE is only used under __CUDACC__, so include-cleaner cannot
// see the use in a non-CUDA parse.
#include "clad/Differentiator/CladConfig.h" // IWYU pragma: keep

#include <cassert>
#include <cstdint>
#include <cstring>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#ifndef Max_Records
#define Max_Records 64
#endif
#ifndef Max_Bytes
#define Max_Bytes 1024
#endif

namespace clad {

/// This class is used for bitwise storing/restoring variables.
/// It is passed to reverse_forw to store the state of the program before the
/// function call. restore() is non-destructive so it can re-establish the
/// pre-call state both before the pullback (whose forward replay must start
/// from it) and after it (the replay re-mutates the restored state):
/// f_reverse_forw(..., _tracker0);
/// ...
/// _tracker0.restore();
/// f_pullback(...);
/// _tracker0.restore();
/// We use it when we have to pass information between nested calls and
/// clad::tape is not viable.
class restore_tracker {
  // m_data consists of pairs of memory addresses and bitwise values
#ifdef __CUDACC__
  struct MetaData {
    char* addr;
    size_t size;
    size_t off;
  };
  MetaData m_meta[Max_Records];
  uint8_t m_buf[Max_Bytes];
  size_t m_cnt = 0, m_off = 0;

public:
  CUDA_HOST_DEVICE restore_tracker() = default;

  template <typename T> CUDA_HOST_DEVICE void store(const T& val) {
    for (size_t i = 0; i < m_cnt; ++i)
      if (m_meta[i].addr == (char*)&val)
        return;

    if (m_cnt >= Max_Records || m_off + sizeof(T) > Max_Bytes) {
      // Clad restore_tracker GPU capacity exceeded. Try again with larger value
      return;
    }

    m_meta[m_cnt] = {(char*)&val, sizeof(T), m_off};
    std::memcpy(m_buf + m_off, &val, sizeof(T));
    m_off += sizeof(T);
    m_cnt++;
  }

  CUDA_HOST_DEVICE void restore() {
    for (size_t i = 0; i < m_cnt; ++i)
      std::memcpy(m_meta[i].addr, m_buf + m_meta[i].off, m_meta[i].size);
  }

  // Drop all records without writing anything back. Emitted where a
  // block-local tracker declaration used to (re-)initialize the tracker.
  CUDA_HOST_DEVICE void clear() { m_cnt = m_off = 0; }
#else
  using Address = char*;
  struct Record {
    Address addr;
    std::size_t size;
    std::size_t off;
  };
  std::vector<Record> m_meta;
  std::vector<uint8_t> m_buf;
  std::unordered_map<Address, std::size_t> m_index;

public:
  // Store the value and the address of `val`.
  template <typename T> void store(const T& val) {
    // If a variable is stored multiple times, we should only take the first
    // value into consideration:
    // _tracker.store(x); // stored
    // ...
    // _tracker.store(x); // ignored
    Address addr = (char*)&val;
    if (!m_index.emplace(addr, m_meta.size()).second)
      return;
    std::size_t off = m_buf.size();
    m_buf.resize(off + sizeof(T));
    std::memcpy(m_buf.data() + off, &val, sizeof(T));
    m_meta.push_back({addr, sizeof(T), off});
  }
  // Set all stored addresses to the corresponding values bitwise. Keeps the
  // stored values: the reverse sweep restores the same state again after the
  // pullback's forward replay re-mutates it.
  void restore() {
    for (const Record& rec : m_meta)
      std::memcpy(rec.addr, m_buf.data() + rec.off, rec.size);
  }

  // Drop all records without writing anything back. Emitted where a
  // block-local tracker declaration used to (re-)initialize the tracker.
  void clear() {
    m_meta.clear();
    m_buf.clear();
    m_index.clear();
  }
#endif
};
} // namespace clad

#endif // CLAD_DIFFERENTIATOR_RESTORETRACKER_H
