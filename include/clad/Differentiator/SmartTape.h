#ifndef CLAD_SMART_TAPE_H
#define CLAD_SMART_TAPE_H

#include "clad/Differentiator/CladConfig.h"
#include <cassert>
#include <cstddef>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace clad {
class smart_tape {
  std::map<void*, std::vector<uint8_t>> _data;
  std::set<std::pair<void*, size_t>> _local_memory;

public:
  template <typename T> void store(const T& val) {
    for (auto& pair : _local_memory)
      if ((char*)pair.first <= (char*)&val &&
          (char*)&val < (char*)((char*)pair.first + pair.second))
        return;
    // Don't overwrite
    if (_data.find((void*)&val) != _data.end())
      return;
    std::vector<uint8_t> buffer(sizeof(T));
    std::memcpy(buffer.data(), &val, sizeof(T));
    _data.emplace((void*)&val, std::move(buffer));
  }
  void restore() {
    for (auto& pair : _data) {
      std::vector<uint8_t>& buffer = pair.second;
      std::memcpy(pair.first, buffer.data(), buffer.size());
    }
    _data.clear();
    _local_memory.clear();
  }
  void reserve(void* loc, size_t size) { _local_memory.emplace(loc, size); }
  template <typename T> void reserve(const T& val) {
    _local_memory.emplace((void*)&val, sizeof(T));
  }
};
} // namespace clad

#endif // CLAD_SMART_TAPE_H
