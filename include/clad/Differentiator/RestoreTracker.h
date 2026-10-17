#ifndef CLAD_DIFFERENTIATOR_RESTORETRACKER_H
#define CLAD_DIFFERENTIATOR_RESTORETRACKER_H

// CUDA_HOST_DEVICE is only used under __CUDACC__, so include-cleaner cannot
// see the use in a non-CUDA parse.
#include "clad/Differentiator/CladConfig.h" // IWYU pragma: keep

#include <cassert>
#include <cstdint>
#include <cstring>
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

  // Store addresses, sizes, and original bytes in one contiguous buffer.
  std::vector<uint8_t> m_data;

public:
  template <typename T> void store(const T& val) {
    const Address addr = (char*)&val;
    size_t offset = 0;

    // Preserve the first value stored for each address.
    while (offset < m_data.size()) {
      Address storedAddr;
      size_t storedSize;

      std::memcpy(&storedAddr, m_data.data() + offset, sizeof(storedAddr));
      offset += sizeof(storedAddr);

      std::memcpy(&storedSize, m_data.data() + offset, sizeof(storedSize));
      offset += sizeof(storedSize);

      if (storedAddr == addr)
        return;

      offset += storedSize;
    }

    const size_t oldSize = m_data.size();
    const size_t valueSize = sizeof(T);
    const size_t recordSize =
        sizeof(addr) + sizeof(valueSize) + valueSize;

    m_data.resize(oldSize + recordSize);

    std::memcpy(m_data.data() + oldSize, &addr, sizeof(addr));
    std::memcpy(m_data.data() + oldSize + sizeof(addr), &valueSize,
                sizeof(valueSize));
    std::memcpy(m_data.data() + oldSize + sizeof(addr) + sizeof(valueSize),
                &val, valueSize);
  }

  void restore() {
    size_t offset = 0;

    while (offset < m_data.size()) {
      Address addr;
      size_t size;

      std::memcpy(&addr, m_data.data() + offset, sizeof(addr));
      offset += sizeof(addr);

      std::memcpy(&size, m_data.data() + offset, sizeof(size));
      offset += sizeof(size);

      std::memcpy(addr, m_data.data() + offset, size);
      offset += size;
    }
  }

  void clear() { m_data.clear(); }
#endif
};
} // namespace clad

#endif // CLAD_DIFFERENTIATOR_RESTORETRACKER_H
