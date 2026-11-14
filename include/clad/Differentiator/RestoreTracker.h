#ifndef CLAD_DIFFERENTIATOR_RESTORETRACKER_H
#define CLAD_DIFFERENTIATOR_RESTORETRACKER_H

// CUDA_HOST_DEVICE is only used under __CUDACC__, so include-cleaner cannot
// see the use in a non-CUDA parse.
#include "clad/Differentiator/CladConfig.h" // IWYU pragma: keep

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <new>
#include <type_traits>
#include <unordered_map>
#include <utility>
#ifndef Max_Records
#define Max_Records 64
#endif
#ifndef Max_Bytes
#define Max_Bytes 1024
#endif

namespace clad {

#ifndef __CUDACC__
namespace detail {
/// A vector of trivially copyable values that stores the first N elements
/// inline and allocates only past them. Most trackers hold one or two small
/// records, for which a heap allocation costs more than the data.
template <typename T, std::size_t N> class inline_vector {
  // NOLINTNEXTLINE(modernize-type-traits): the headers still build as C++14.
  static_assert(std::is_trivially_copyable<T>::value,
                "elements are moved with memcpy");
  std::array<T, N> m_Inline;
  T* m_Data = m_Inline.data();
  std::size_t m_Size = 0;
  std::size_t m_Capacity = N;

  [[nodiscard]] bool isInline() const { return m_Data == m_Inline.data(); }

  // realloc can grow a large buffer in place or remap its pages instead of
  // copying it. A tracker that saves a whole array grows to the array's size.
  // NOLINTBEGIN(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  CLAD_NOINLINE void grow(std::size_t size) {
    std::size_t capacity = 2 * m_Capacity < size ? size : 2 * m_Capacity;
    void* data = nullptr;
    if (isInline()) {
      data = std::malloc(capacity * sizeof(T));
      if (data)
        std::memcpy(data, m_Data, m_Size * sizeof(T));
    } else {
      data = std::realloc(m_Data, capacity * sizeof(T));
    }
    if (!data)
#ifdef __cpp_exceptions
      throw std::bad_alloc();
#else
      std::abort();
#endif
    m_Data = static_cast<T*>(data);
    m_Capacity = capacity;
  }
  /// Inline: most trackers stay within the inline buffer, and then releasing
  /// is one compare.
  void release() {
    if (!isInline())
      freeHeap();
  }
  CLAD_NOINLINE void freeHeap() { std::free(m_Data); }
  // NOLINTEND(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)

public:
  // The inline elements are left uninitialized; each is written before it is
  // read.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init,modernize-use-equals-default)
  inline_vector() {}
  // Trackers are pushed onto tapes by copy and popped back.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
  inline_vector(const inline_vector& other) { *this = other; }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
  inline_vector(inline_vector&& other) noexcept { *this = std::move(other); }
  inline_vector& operator=(const inline_vector& other) {
    if (this != &other) {
      resize(other.m_Size);
      std::memcpy(m_Data, other.m_Data, m_Size * sizeof(T));
    }
    return *this;
  }
  inline_vector& operator=(inline_vector&& other) noexcept {
    if (this != &other) {
      if (other.isInline()) {
        // other's elements fit in the inline buffer: copy them without
        // allocating.
        m_Size = other.m_Size;
        std::memcpy(m_Data, other.m_Data, m_Size * sizeof(T));
      } else {
        release();
        m_Data = other.m_Data;
        m_Capacity = other.m_Capacity;
        m_Size = other.m_Size;
        other.m_Data = other.m_Inline.data();
        other.m_Capacity = N;
      }
      other.m_Size = 0;
    }
    return *this;
  }
  ~inline_vector() { release(); }

  [[nodiscard]] std::size_t size() const { return m_Size; }
  [[nodiscard]] bool empty() const { return m_Size == 0; }
  T* data() { return m_Data; }
  // The part of std::vector's interface the records use. m_Data holds m_Size
  // elements.
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  T* begin() { return m_Data; }
  T* end() { return m_Data + m_Size; }
  std::reverse_iterator<T*> rbegin() {
    return std::reverse_iterator<T*>(end());
  }
  std::reverse_iterator<T*> rend() {
    return std::reverse_iterator<T*>(begin());
  }
  T& operator[](std::size_t i) { return m_Data[i]; }
  T& back() { return m_Data[m_Size - 1]; }
  void push_back(const T& value) {
    if (m_Size == m_Capacity)
      grow(m_Size + 1);
    m_Data[m_Size++] = value;
  }
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  /// Grows without initializing the new elements.
  void resize(std::size_t size) {
    if (size > m_Capacity)
      grow(size);
    m_Size = size;
  }
  void clear() { m_Size = 0; }
};
} // namespace detail
#endif

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
  // m_meta holds each stored address with its size and offset into m_buf.
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
    // Skip the store only if a record already covers all of val: a member
    // stored before the object around it shares the object's address but
    // not its size.
    auto* addr = (char*)&val;
    for (size_t i = 0; i < m_cnt; ++i)
      if (addr >= m_meta[i].addr &&
          addr + sizeof(T) <= m_meta[i].addr + m_meta[i].size)
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

  // Newest first, so where records overlap the oldest snapshot wins.
  CUDA_HOST_DEVICE void restore() {
    for (size_t i = m_cnt; i-- > 0;)
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
  // Most calls store a few scalars in one record. One inline record and 16
  // inline bytes cover them without allocating, and the tracker is smaller
  // than with two std::vectors.
  // NOLINTNEXTLINE(readability-identifier-naming): named as it was.
  detail::inline_vector<Record, 1> m_meta;
  // NOLINTNEXTLINE(readability-identifier-naming): named like m_meta.
  detail::inline_vector<uint8_t, 16> m_buf;
  // Past kScanLimit records, the widest store made at each address the index
  // knows: every record's start when it is built, and every store looked up
  // after that. A member stored before the object around it shares the
  // object's address but not its size, so the address alone would take the
  // object for a repeat and drop the rest of it.
  // Built on first use and dropped on copy; it is rebuilt from m_meta.
  using Index = std::unordered_map<Address, std::size_t>;
  /// Deletes the index out of line, so that destroying a tracker without an
  /// index stays small enough to inline into the derivative.
  struct IndexDeleter {
    CLAD_NOINLINE void operator()(Index* index) const {
      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      delete index;
    }
  };
  // NOLINTNEXTLINE(readability-identifier-naming): named like m_meta.
  std::unique_ptr<Index, IndexDeleter> m_index;
  static constexpr std::size_t kScanLimit = 16;

  // Is all of [addr, addr + size) recorded already? While the records are
  // few this is exact. Past them, a store that starts inside a record but at
  // none of the addresses the index knows is recorded again. That costs a
  // record, not correctness: restore() writes the records newest first, so
  // every byte ends with the oldest value stored for it.
  bool known(Address addr, std::size_t size) {
    if (m_meta.size() <= kScanLimit) {
      std::less<> before;
      for (const Record& r : m_meta)
        if (!before(addr, r.addr) && !before(r.addr + r.size, addr + size))
          return true;
      return false;
    }
    // emplace rather than operator[]: clang 17 cannot instantiate the tuple
    // operator[] builds with libstdc++ 14 in C++23.
    if (!m_index) {
      m_index.reset(new Index()); // NOLINT(cppcoreguidelines-owning-memory)
      for (const Record& r : m_meta) {
        std::size_t& widest = m_index->emplace(r.addr, 0).first->second;
        widest = widest < r.size ? r.size : widest;
      }
    }
    std::size_t& widest = m_index->emplace(addr, 0).first->second;
    if (size <= widest)
      return true;
    widest = size;
    return false;
  }

  CLAD_NOINLINE void storeSlow(Address addr, std::size_t size) {
    if (known(addr, size))
      return;
    std::size_t off = m_buf.size();
    m_buf.resize(off + size);
    std::memcpy(m_buf.data() + off, addr, size);
    // A callee that overwrites a buffer stores it one element at a time, in
    // order: grow the last record rather than adding one per element.
    if (!m_meta.empty() && m_meta.back().addr + m_meta.back().size == addr) {
      m_meta.back().size += size;
      return;
    }
    m_meta.push_back({addr, size, off});
  }

public:
  // User-provided, so that `restore_tracker t = {};` in a derivative does not
  // zero the inline storage first.
  // NOLINTNEXTLINE(modernize-use-equals-default)
  restore_tracker() {}
  restore_tracker(const restore_tracker& other)
      : m_meta(other.m_meta), m_buf(other.m_buf) {}
  restore_tracker& operator=(const restore_tracker& other) {
    if (this != &other) {
      m_meta = other.m_meta;
      m_buf = other.m_buf;
      m_index.reset();
    }
    return *this;
  }
  restore_tracker(restore_tracker&&) noexcept = default;
  restore_tracker& operator=(restore_tracker&&) noexcept = default;
  ~restore_tracker() = default;

  // Store the value and the address of `val`.
  template <typename T> void store(const T& val) {
    // If a variable is stored multiple times, we should only take the first
    // value into consideration:
    // _tracker.store(x); // stored
    // ...
    // _tracker.store(x); // ignored
    Address addr = (char*)&val;
    // The common case, inline: a single record so far, and val starts right
    // after it, so no record can cover it yet.
    if (m_meta.size() == 1 && m_meta[0].addr + m_meta[0].size == addr) {
      std::size_t off = m_buf.size();
      m_buf.resize(off + sizeof(T));
      std::memcpy(m_buf.data() + off, addr, sizeof(T));
      m_meta[0].size += sizeof(T);
      return;
    }
    storeSlow(addr, sizeof(T));
  }
  // Set all stored addresses to the corresponding values bitwise. Keeps the
  // stored values: the reverse sweep restores the same state again after the
  // pullback's forward replay re-mutates it. Newest first, so where records
  // overlap the oldest snapshot wins.
  void restore() {
    for (auto it = m_meta.rbegin(); it != m_meta.rend(); ++it)
      std::memcpy(it->addr, m_buf.data() + it->off, it->size);
  }

  // Drop all records without writing anything back. Emitted where a
  // block-local tracker declaration used to (re-)initialize the tracker.
  void clear() {
    m_meta.clear();
    m_buf.clear();
    m_index.reset();
  }
#endif
};
} // namespace clad

#endif // CLAD_DIFFERENTIATOR_RESTORETRACKER_H
