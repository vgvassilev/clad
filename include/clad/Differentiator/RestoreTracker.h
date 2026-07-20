#ifndef CLAD_DIFFERENTIATOR_RESTORETRACKER_H
#define CLAD_DIFFERENTIATOR_RESTORETRACKER_H

#include <cassert>
#include <cstdint>
#include <cstring>
#include <map>
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
/// function call, for example,
/// f_reverse_forw(..., _tracker0);
/// ...
/// _tracker0.restore();
/// f_pullback(...);
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
  __host__ __device__ restore_tracker() = default;

  template <typename T> __host__ __device__ void store(const T& val) {
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

  __host__ __device__ void restore() {
    for (size_t i = 0; i < m_cnt; ++i)
      std::memcpy(m_meta[i].addr, m_buf + m_meta[i].off, m_meta[i].size);
    m_cnt = m_off = 0;
  }
#else
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
    for (std::pair<const Address, RawMemory>& pair : m_data) {
      std::vector<uint8_t>& buffer = pair.second;
      std::memcpy(pair.first, buffer.data(), buffer.size());
    }
    m_data.clear();
  }
#endif
};
} // namespace clad

#endif // CLAD_DIFFERENTIATOR_RESTORETRACKER_H
