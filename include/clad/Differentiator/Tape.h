#ifndef CLAD_TAPE_H
#define CLAD_TAPE_H

#include "clad/Differentiator/CladConfig.h"
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <iterator>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace clad {

template <typename T, std::size_t SBO_SIZE, std::size_t SLAB_SIZE>
class tape_impl;

/// A forward iterator for traversing elements in `clad::tape_impl`.
/// This iterator supports standard forward iteration operations, including:
/// - Dereferencing (`*`, `->`)
/// - Increment (`++`)
/// - Equality and inequality comparisons
template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024>
class tape_iterator {
  using tape_t = clad::tape_impl<T, SBO_SIZE, SLAB_SIZE>;
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
template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024>
class tape_impl {
  /// A block of contiguous storage allocated dynamically when SBO capacity is
  /// exceeded.
  struct Slab {
    // std::aligned_storage_t<sizeof(T), alignof(T)> raw_data[SLAB_SIZE];
    // For now use the implementation below as above implementation is not
    // supported by c++11
    alignas(T) char raw_data[SLAB_SIZE * sizeof(T)]{};
    Slab* next;
    CUDA_HOST_DEVICE Slab() : next(nullptr) {}
    CUDA_HOST_DEVICE T* elements() {
#if __cplusplus >= 201703L
      return std::launder(reinterpret_cast<T*>(raw_data));
#else
      return reinterpret_cast<T*>(raw_data);
#endif
    }
  };

  // std::aligned_storage_t<sizeof(T), alignof(T)> m_static_buffer[SBO_SIZE];
  // For now use the implementation below as above implementation is not
  // supported by c++11
  alignas(T) char m_static_buffer[SBO_SIZE * sizeof(T)]{};
  bool m_using_sbo = true;

  Slab* m_head = nullptr;
  Slab* m_tail = nullptr;
  std::size_t m_size = 0;
  std::size_t m_capacity = SBO_SIZE;

  CUDA_HOST_DEVICE T* sbo_elements() {
#if __cplusplus >= 201703L
    return std::launder(reinterpret_cast<T*>(m_static_buffer));
#else
    return reinterpret_cast<T*>(m_static_buffer);
#endif
  }

  CUDA_HOST_DEVICE const T* sbo_elements() const {
#if __cplusplus >= 201703L
    return std::launder(reinterpret_cast<const T*>(m_static_buffer));
#else
    return reinterpret_cast<const T*>(m_static_buffer);
#endif
  }

public:
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using iterator = tape_iterator<T, SBO_SIZE, SLAB_SIZE>;
  using const_iterator = tape_iterator<const T, SBO_SIZE, SLAB_SIZE>;

  CUDA_HOST_DEVICE tape_impl() = default;

  CUDA_HOST_DEVICE ~tape_impl() { clear(); }

  /// Add new value of type T constructed from args to the end of the tape.
  template <typename... ArgsT>
  CUDA_HOST_DEVICE void emplace_back(ArgsT&&... args) {
    if (m_size < SBO_SIZE) {
      // Store in SBO buffer
      ::new (const_cast<void*>(static_cast<const volatile void*>(
          sbo_elements() + m_size))) T(std::forward<ArgsT>(args)...);
    } else {
      // Transition to dynamic storage if needed
      if (m_using_sbo)
        m_using_sbo = false;

      // Allocate new slab if required
      if (m_size == m_capacity) {
        Slab* new_slab = new Slab();
        if (!m_head)
          m_head = new_slab;
        else
          m_tail->next = new_slab;
        m_tail = new_slab;
        m_capacity += SLAB_SIZE;
      } else if ((m_size - SBO_SIZE) % SLAB_SIZE == 0) {
        if (m_size == SBO_SIZE)
          m_tail = m_head;
        else
          m_tail = m_tail->next;
      }

      // Construct element in-place
      ::new (const_cast<void*>(static_cast<const volatile void*>(
          m_tail->elements() + ((m_size - SBO_SIZE) % SLAB_SIZE))))
          T(std::forward<ArgsT>(args)...);
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
    return (*this)[m_size - 1];
  }

  CUDA_HOST_DEVICE const_reference back() const {
    assert(m_size);
    return (*this)[m_size - 1];
  }

  CUDA_HOST_DEVICE reference operator[](std::size_t i) {
    assert(i < m_size);
    return *at(i);
  }

  CUDA_HOST_DEVICE const_reference operator[](std::size_t i) const {
    assert(i < m_size);
    return *at(i);
  }

  /// Remove the last value from the tape.
  CUDA_HOST_DEVICE void pop_back() {
    assert(m_size);
    m_size--;
    if (m_size < SBO_SIZE)
      destroy_element(sbo_elements() + m_size);
    else {
      std::size_t offset = (m_size - SBO_SIZE) % SLAB_SIZE;
      destroy_element(m_tail->elements() + offset);
      if (offset == 0) {
        Slab* slab = m_head;
        Slab* prev = m_head;
        std::size_t idx = (m_size - SBO_SIZE) / SLAB_SIZE;
        while (idx--) {
          prev = slab;
          slab = slab->next;
        }
        m_tail = prev;
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
    return slab->elements() + ((index - SBO_SIZE) % SLAB_SIZE);
  }

  CUDA_HOST_DEVICE const T* at(std::size_t index) const {
    if (index < SBO_SIZE)
      return sbo_elements() + index;
    Slab* slab = m_head;
    std::size_t idx = (index - SBO_SIZE) / SLAB_SIZE;
    while (idx--)
      slab = slab->next;
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
  void clear() {
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
      delete tmp;
    }

    m_head = nullptr;
    m_tail = nullptr;
    m_size = 0;
    m_capacity = SBO_SIZE;
    m_using_sbo = true;
  }

  template <typename ElTy> void destroy_element(ElTy* elem) { elem->~ElTy(); }

  template <typename ElTy, size_t N> void destroy_element(ElTy (*arr)[N]) {
    for (size_t i = 0; i < N; ++i)
      (*arr)[i].~ElTy();
  }
};
} // namespace clad

#endif // CLAD_TAPE_H