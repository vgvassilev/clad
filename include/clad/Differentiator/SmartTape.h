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
  struct interval {
    char *min = nullptr, *max = nullptr;

    interval(char* pmin = nullptr, char* pmax = nullptr)
        : min(pmin), max(pmax) {
      assert((!max || min <= max) && "negative length interval");
    }

    inline char compr_impl(const interval& other) const {
      if (this->is_single_element()) {
        if (this->min < other.min)
          return -1;
        char* other_max = other.max ? other.max : (other.min + 1);
        if (this->min >= other_max)
          return 1;
        return 0;
      }
      if (other.is_single_element()) {
        if (other.min < this->min)
          return 1;
        if (other.min >= this->max)
          return -1;
        return 0;
      }
      if (this->min == other.min) {
        assert(this->max == other.max && "comparing overlapping intervals");
        return 0;
      }
      if (this->min < other.min) {
        assert(this->max <= other.min && "comparing overlapping intervals");
        return -1;
      }
      // else (this->min > other.min)
      assert(this->min >= other.max && "comparing overlapping intervals");
      return 1;
    }

    bool operator==(const interval& other) const {
      return this->compr_impl(other) == 0;
    }

    bool operator>(const interval& other) const {
      return this->compr_impl(other) == 1;
    }

    bool operator<(const interval& other) const {
      return this->compr_impl(other) == -1;
    }

    inline bool is_single_element() const { return max == nullptr; }
  };
  std::map<const interval, std::vector<uint8_t>> _data;

public:
  template <typename T> void store(const T& val) {
    // Don't overwrite
    if (_data.find({(char*)&val}) != _data.end())
      return;
    std::vector<uint8_t> buffer(sizeof(T) + 1);
    buffer[0] = 1;
    std::memcpy(buffer.data() + 1, &val, sizeof(T));
    _data.emplace(interval{(char*)&val}, std::move(buffer));
  }
  void restore() {
    for (auto& pair : _data) {
      std::vector<uint8_t>& buffer = pair.second;
      if (buffer[0])
        std::memcpy(pair.first.min, buffer.data() + 1, buffer.size() - 1);
    }
    _data.clear();
  }
  void ignore(void* loc, size_t size) {
    _data.emplace(interval{(char*)loc, (char*)loc + size},
                  std::vector<uint8_t>{0});
  }
  template <typename T> void ignore(const T& val) {
    _data.emplace(interval{(char*)&val, (char*)&val + sizeof(T)},
                  std::vector<uint8_t>{0});
  }
};
} // namespace clad

#endif // CLAD_SMART_TAPE_H
