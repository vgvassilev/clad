#ifndef CLAD_TAPE_H
#define CLAD_TAPE_H

#include "clad/Differentiator/CladConfig.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#ifndef __CUDACC__
#include <mutex>
#endif

namespace clad {

namespace detail {

/// Manages offloading of data to disk when RAM capacity is exceeded.
/// Handles files I/O operations for reading and writing slabs.
template <typename T, std::size_t SLAB_SIZE> struct DiskManager {
#ifndef __CUDA_ARCH__
  std::fstream file;
  std::string filename;
  DiskManager() {
    filename = "clad_tape_" + std::to_string((uintptr_t)this) + ".tmp";
    file.open(filename, std::ios::in | std::ios::out | std::ios::binary |
                            std::ios::trunc);
  }
  ~DiskManager() {
    if (file.is_open())
      file.close();
    std::remove(filename.c_str());
  }
  std::size_t write_slab(const T* data) {
    file.seekp(0, std::ios::end);
    std::size_t offset = file.tellp();
    const void* raw_data = static_cast<const void*>(data);
    file.write(static_cast<const char*>(raw_data), SLAB_SIZE * sizeof(T));
    return offset;
  }
  void read_slab(T* dest, std::size_t offset) {
    file.seekg(offset, std::ios::beg);
    void* raw_dest = static_cast<void*>(dest);
    file.read(static_cast<char*>(raw_dest), SLAB_SIZE * sizeof(T));
  }
#else
  CUDA_HOST_DEVICE DiskManager() {}
  CUDA_HOST_DEVICE ~DiskManager() {}
  CUDA_HOST_DEVICE std::size_t write_slab(const T* data) { return 0; }
  CUDA_HOST_DEVICE void read_slab(T* dest, std::size_t offset) {}
#endif
};

struct NoOpMutex {
  void lock() {}
  void unlock() {}
  static bool try_lock() { return true; }
};

} // namespace detail

template <typename T, std::size_t SBO_SIZE, std::size_t SLAB_SIZE,
          bool is_multithread, bool DiskOffload>
class tape_impl;

/// A forward iterator for traversing elements in `clad::tape_impl`.
/// This iterator supports standard forward iteration operations, including:
/// - Dereferencing (`*`, `->`)
/// - Increment (`++`)
/// - Equality and inequality comparisons
template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024,
          bool is_multithread = false, bool DiskOffload = false>
class tape_iterator {
  using tape_t =
      clad::tape_impl<T, SBO_SIZE, SLAB_SIZE, is_multithread, DiskOffload>;
  tape_t* m_tape;
  std::size_t m_index;

public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  CUDA_HOST_DEVICE tape_iterator() : m_tape(nullptr), m_index(0) {}
  CUDA_HOST_DEVICE tape_iterator(tape_t* tape, std::size_t index)
      : m_tape(tape), m_index(index) {}

  CUDA_HOST_DEVICE reference operator*() const { return (*m_tape)[m_index]; }

  CUDA_HOST_DEVICE pointer operator->() const { return &(*m_tape)[m_index]; }

  CUDA_HOST_DEVICE tape_iterator& operator++() {
    ++m_index;
    return *this;
  }

  CUDA_HOST_DEVICE tape_iterator operator++(int) {
    tape_iterator tmp = *this;
    ++(*this);
    return tmp;
  }

  CUDA_HOST_DEVICE bool operator==(const tape_iterator& other) const {
    return m_index == other.m_index;
  }

  CUDA_HOST_DEVICE bool operator!=(const tape_iterator& other) const {
    return m_index != other.m_index;
  }
};

/// A dynamic slab-based vector-like container with Small Buffer Optimization
/// (SBO), primarily used for storing values in reverse-mode AD. Stores elements
/// in a static buffer first, then falls back to dynamically allocated linked
/// slabs if capacity exceeds SBO.
template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024,
          bool is_multithread = false, bool DiskOffload = false>
class tape_impl {
  /// Storage planning for slabs kept in memory (RAM).
  /// Provides access to the raw data buffer.
  struct RAMStorage {
    // std::aligned_storage_t<sizeof(T), alignof(T)> raw_data[SLAB_SIZE];
    // For now use the implementation below as above implementation is not
    // supported by c++11
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    alignas(T) char raw_data[SLAB_SIZE * sizeof(T)];
    CUDA_HOST_DEVICE RAMStorage() {}
    CUDA_HOST_DEVICE T* elements() {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
      return reinterpret_cast<T*>(raw_data);
    }
  };

  struct DiskStorage {
    T* data_ptr = nullptr;
    bool is_on_disk = false;
    std::size_t disk_offset = 0;

    CUDA_HOST_DEVICE DiskStorage() { allocate(); }
    CUDA_HOST_DEVICE ~DiskStorage() { deallocate(); }

    DiskStorage(const DiskStorage&) = delete;
    DiskStorage& operator=(const DiskStorage&) = delete;

    void allocate() {
      if (!data_ptr)
        data_ptr = static_cast<T*>(::operator new(SLAB_SIZE * sizeof(T)));
    }
    void deallocate() {
      if (data_ptr) {
        ::operator delete(data_ptr);
        data_ptr = nullptr;
      }
    }
    CUDA_HOST_DEVICE T* elements() { return data_ptr; }
  };

  using SlabBase =
      typename std::conditional<DiskOffload, DiskStorage, RAMStorage>::type;

public:
  /// A block of contiguous storage allocated dynamically when SBO capacity is
  /// exceeded.
  struct Slab : public SlabBase {
    Slab* prev;
    Slab* next;
    CUDA_HOST_DEVICE Slab() : prev(nullptr), next(nullptr) {}
  };

private:
  // std::aligned_storage_t<sizeof(T), alignof(T)> m_static_buffer[SBO_SIZE];
  // For now use the implementation below as above implementation is not
  // supported by c++11
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  alignas(T) char m_static_buffer[SBO_SIZE * sizeof(T)];

  Slab* m_head = nullptr;
  Slab* m_tail = nullptr;
  std::size_t m_size = 0;
  std::size_t m_capacity = SBO_SIZE;
#ifndef __CUDACC__
  mutable std::mutex m_TapeMutex;
#endif
  /// Holds current state related to disk offloading, including the file manager
  /// and also keep track of active/maximum RAM slabs.
  struct DiskInfo {
    std::unique_ptr<detail::DiskManager<T, SLAB_SIZE>> m_DiskManager;
    std::size_t m_ActiveSlabs = 0;
    std::size_t m_MaxRamSlabs = 1024;
    DiskInfo() = default;
  };
  struct Empty {};

  using DiskInfoType = std::conditional_t<DiskOffload, DiskInfo, Empty>;
  // NOLINTNEXTLINE(readability-identifier-naming)
  DiskInfoType m_state;

  CUDA_HOST_DEVICE T* sbo_elements() {
#if __cplusplus >= 201703L
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return std::launder(reinterpret_cast<T*>(m_static_buffer));
#else
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<T*>(m_static_buffer);
#endif
  }

  CUDA_HOST_DEVICE const T* sbo_elements() const {
#if __cplusplus >= 201703L
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return std::launder(reinterpret_cast<const T*>(m_static_buffer));
#else
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<const T*>(m_static_buffer);
#endif
  }

  DiskInfo& getDiskInfo() {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return *reinterpret_cast<DiskInfo*>(&m_state);
  }

  void check_and_evict_impl(std::true_type) {
    DiskInfo& info = getDiskInfo();
    if (info.m_ActiveSlabs >= info.m_MaxRamSlabs) {
      Slab* candidate = m_head;
      while (candidate && (candidate->is_on_disk || candidate == m_tail))
        candidate = candidate->next;
      if (candidate) {
        if (!info.m_DiskManager)
          info.m_DiskManager.reset(new detail::DiskManager<T, SLAB_SIZE>());
        candidate->disk_offset =
            info.m_DiskManager->write_slab(candidate->elements());
        // FIXME: We probably should not deallocate the slab but use a pool
        // where old ones would be recycled.
        candidate->deallocate();
        candidate->is_on_disk = true;
        info.m_ActiveSlabs--;
      }
    }
  }

  void check_and_evict_impl(std::false_type) {}

  void ensure_loaded_impl(Slab* slab, std::true_type) {
    if (slab && slab->is_on_disk) {
      DiskInfo& info = getDiskInfo();
      if (info.m_ActiveSlabs >= info.m_MaxRamSlabs) {
        Slab* v = m_head;
        while (v) {
          if (!v->is_on_disk && v != slab) {
            v->disk_offset = info.m_DiskManager->write_slab(v->elements());
            v->deallocate();
            v->is_on_disk = true;
            info.m_ActiveSlabs--;
            break;
          }
          v = v->next;
        }
      }
      slab->allocate();
      info.m_DiskManager->read_slab(slab->elements(), slab->disk_offset);
      slab->is_on_disk = false;
      info.m_ActiveSlabs++;
    }
  }

  void ensure_loaded_impl(Slab* slab, std::false_type) {}

  void check_and_evict() {
    check_and_evict_impl(std::integral_constant<bool, DiskOffload>{});
  }

  void ensure_loaded(Slab* slab) {
    ensure_loaded_impl(slab, std::integral_constant<bool, DiskOffload>{});
  }

public:
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using iterator =
      tape_iterator<T, SBO_SIZE, SLAB_SIZE, is_multithread, DiskOffload>;
  using const_iterator =
      tape_iterator<const T, SBO_SIZE, SLAB_SIZE, is_multithread, DiskOffload>;
#ifndef __CUDACC__

  std::mutex& mutex() const { return m_TapeMutex; }

#endif
  CUDA_HOST_DEVICE tape_impl() = default;

  CUDA_HOST_DEVICE ~tape_impl() { clear(); }

  tape_impl(const tape_impl&) = delete;
  tape_impl& operator=(const tape_impl&) = delete;

  tape_impl(tape_impl&& other) = delete;
  tape_impl& operator=(tape_impl&& other) = delete;

  /// Add new value of type T constructed from args to the end of the tape.
  template <typename... ArgsT>
  CUDA_HOST_DEVICE void emplace_back(ArgsT&&... args) {
    if (m_size < SBO_SIZE) {
      // Store in SBO buffer
      ::new (const_cast<void*>(static_cast<const volatile void*>(
          sbo_elements() + m_size))) T(std::forward<ArgsT>(args)...);
    } else {
      const auto offset = (m_size - SBO_SIZE) % SLAB_SIZE;
      // Allocate new slab if required
      if (!offset) {
        if (m_size == m_capacity) {
          check_and_evict();

          Slab* new_slab = new Slab();
          if (DiskOffload)
            getDiskInfo().m_ActiveSlabs++;

          if (!m_head)
            m_head = new_slab;
          else {
            m_tail->next = new_slab;
            new_slab->prev = m_tail;
          }
          m_capacity += SLAB_SIZE;
        }
        if (m_size == SBO_SIZE)
          m_tail = m_head;
        else
          m_tail = m_tail->next;
      }

      // Construct element in-place
      if (DiskOffload)
        ensure_loaded(m_tail);
      ::new (const_cast<void*>(static_cast<const volatile void*>(
          m_tail->elements() + offset))) T(std::forward<ArgsT>(args)...);
    }
    m_size++;
  }

  CUDA_HOST_DEVICE std::size_t size() const { return m_size; }

  CUDA_HOST_DEVICE iterator begin() { return iterator(this, 0); }

  CUDA_HOST_DEVICE const_iterator begin() const {
    return const_iterator(this, 0);
  }

  CUDA_HOST_DEVICE iterator end() { return iterator(this, m_size); }

  CUDA_HOST_DEVICE const_iterator end() const {
    return const_iterator(this, m_size);
  }

  /// Access last value (must not be empty).
  CUDA_HOST_DEVICE reference back() {
    assert(m_size);
    std::size_t index = m_size - 1;
    if (index < SBO_SIZE)
      return *(sbo_elements() + index);
    if (DiskOffload)
      ensure_loaded(m_tail);
    index = (index - SBO_SIZE) % SLAB_SIZE;
    return *(m_tail->elements() + index);
  }

  CUDA_HOST_DEVICE const_reference back() const {
    assert(m_size);
    std::size_t index = m_size - 1;
    if (index < SBO_SIZE)
      return *(sbo_elements() + index);
    if (DiskOffload) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<tape_impl*>(this)->ensure_loaded(m_tail);
    }
    index = (index - SBO_SIZE) % SLAB_SIZE;
    return *(m_tail->elements() + index);
  }

  CUDA_HOST_DEVICE reference operator[](std::size_t i) {
    assert(i < m_size);
    return *at(i);
  }

  CUDA_HOST_DEVICE const_reference operator[](std::size_t i) const {
    assert(i < m_size);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    return *const_cast<tape_impl*>(this)->at(i);
  }

  /// Remove the last value from the tape.
  CUDA_HOST_DEVICE void pop_back() {
    assert(m_size);
    m_size--;
    if (m_size < SBO_SIZE)
      destroy_element(sbo_elements() + m_size);
    else {
      if (DiskOffload)
        ensure_loaded(m_tail);

      std::size_t offset = (m_size - SBO_SIZE) % SLAB_SIZE;
      destroy_element(m_tail->elements() + offset);
      if (offset == 0) {
        if (m_tail != m_head)
          m_tail = m_tail->prev;
      }
    }
  }

private:
  /// Returns pointer to element at specified index, handling SBO or slab lookup
  CUDA_HOST_DEVICE T* at(std::size_t index) {
    if (index < SBO_SIZE)
      return sbo_elements() + index;

    Slab* slab = m_head;
    std::size_t idx = (index - SBO_SIZE) / SLAB_SIZE;
    while (idx--)
      slab = slab->next;

    if (DiskOffload)
      ensure_loaded(slab);

    return slab->elements() + ((index - SBO_SIZE) % SLAB_SIZE);
  }

  CUDA_HOST_DEVICE const T* at(std::size_t index) const {
    if (index < SBO_SIZE)
      return sbo_elements() + index;
    Slab* slab = m_head;
    std::size_t idx = (index - SBO_SIZE) / SLAB_SIZE;
    while (idx--)
      slab = slab->next;

    // Const version cannot ensure loaded if DiskOffload is true
    return slab->elements() + ((index - SBO_SIZE) % SLAB_SIZE);
  }

  template <typename It> using value_type_of = decltype(*std::declval<It>());

  // Call destructor for every value in the given range.
  template <typename It>
  static typename std::enable_if<
      !std::is_trivially_destructible<value_type_of<It>>::value>::type
  destroy(It B, It E) {
    for (It I = E - 1; I >= B; --I)
      I->~value_type_of<It>();
  }

  // If type is trivially destructible, its destructor is no-op, so we can avoid
  // for loop here.
  template <typename It>
  static typename std::enable_if<
      std::is_trivially_destructible<value_type_of<It>>::value>::type
      CUDA_HOST_DEVICE
      destroy(It B, It E) {}

  /// Destroys all elements and deallocates slabs
  void clear_impl(std::true_type) {
    std::size_t count = m_size;
    for (std::size_t i = 0; i < SBO_SIZE && count > 0; ++i, --count)
      destroy_element(&sbo_elements()[i]);

    Slab* slab = m_head;
    while (slab) {
      size_t current_slab_count = (count > SLAB_SIZE) ? SLAB_SIZE : count;
      count -= current_slab_count;
      Slab* tmp = slab;
      slab = slab->next;
      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      delete tmp;
    }
    getDiskInfo().m_ActiveSlabs = 0;
  }

  void clear_impl(std::false_type) {
    std::size_t count = m_size;
    for (std::size_t i = 0; i < SBO_SIZE && count > 0; ++i, --count)
      destroy_element(&sbo_elements()[i]);

    Slab* slab = m_head;
    while (slab) {
      T* elems = slab->elements();
      for (size_t i = 0; i < SLAB_SIZE && count > 0; ++i, --count)
        destroy_element(elems + i);
      Slab* tmp = slab;
      slab = slab->next;
      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      delete tmp;
    }
  }

  void clear() {
    clear_impl(std::integral_constant<bool, DiskOffload>{});
    m_head = nullptr;
    m_tail = nullptr;
    m_size = 0;
    m_capacity = SBO_SIZE;
  }

  template <typename ElTy> void destroy_element(ElTy* elem) { elem->~ElTy(); }
  template <typename ElTy, size_t N> void destroy_element(ElTy (*arr)[N]) {
    for (size_t i = 0; i < N; ++i)
      (*arr)[i].~ElTy();
  }
};
} // namespace clad

#endif // CLAD_TAPE_H