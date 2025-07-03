#ifndef CLAD_TAPE_H
#define CLAD_TAPE_H

#include <cassert>
#include <cstdio>
#include <memory>
#include <type_traits>
#include <utility>
#include "clad/Differentiator/CladConfig.h"

namespace clad {
  /// A dynamic slab-based vector-like container with Small Buffer Optimization (SBO),
  /// primarily used for storing values in reverse-mode AD.
  /// Stores elements in a static buffer first, then falls back to dynamically
  /// allocated linked slabs if capacity exceeds SBO.
  template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024>
  class tape_impl {
    /// A block of contiguous storage allocated dynamically when SBO capacity is exceeded.
    struct Slab {
      alignas(T) char raw_data[SLAB_SIZE * sizeof(T)]{};
      Slab* next;
      CUDA_HOST_DEVICE Slab() : next(nullptr) {}
      CUDA_HOST_DEVICE T* elements() { return reinterpret_cast<T*>(raw_data); }
    };

    alignas(T) char m_static_buffer[SBO_SIZE * sizeof(T)];
    bool m_using_sbo = true;

    Slab* m_head = nullptr;
    std::size_t _size = 0;

    CUDA_HOST_DEVICE T* sbo_elements() {
      return reinterpret_cast<T*>(m_static_buffer);
    }

    CUDA_HOST_DEVICE const T* sbo_elements() const {
      return reinterpret_cast<const T*>(m_static_buffer);
    }

  public:
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using iterator = pointer;
    using const_iterator = const_pointer;

    CUDA_HOST_DEVICE ~tape_impl() {
      clear();
    }

    /// Add new value of type T constructed from args to the end of the tape.
    template <typename... ArgsT>
    CUDA_HOST_DEVICE void emplace_back(ArgsT&&... args) {
      if (_size < SBO_SIZE) {
        // Store in SBO buffer
        ::new (const_cast<void*>(static_cast<const volatile void*>(sbo_elements() + _size)))
            T(std::forward<ArgsT>(args)...);
      } else {
        // Transition to dynamic storage if needed
        if (m_using_sbo) {
          m_using_sbo = false;
        }

        // Allocate new slab if required
        if ((_size - SBO_SIZE) % SLAB_SIZE == 0) {
          Slab* new_slab = new Slab();
          if (!m_head) {
            m_head = new_slab;
          } else {
            Slab* last = m_head;
            while (last->next) last = last->next;
            last->next = new_slab;
          }
        }

        // Find correct slab for element
        Slab* slab = m_head;
        std::size_t idx = (_size - SBO_SIZE) / SLAB_SIZE;
        while (idx--) slab = slab->next;

        // Construct element in-place
        ::new (const_cast<void*>(static_cast<const volatile void*>(
            slab->elements() + ((_size - SBO_SIZE) % SLAB_SIZE))))
            T(std::forward<ArgsT>(args)...);
      }
      _size++;
    }

    CUDA_HOST_DEVICE std::size_t size() const { return _size; }

    CUDA_HOST_DEVICE pointer begin() {
      return sbo_elements();
    }

    CUDA_HOST_DEVICE const_pointer begin() const {
      return sbo_elements();
    }

    CUDA_HOST_DEVICE pointer end() {
      return at(_size);
    }

    CUDA_HOST_DEVICE const_pointer end() const {
      return at(_size);
    }

    /// Access last value (must not be empty).
    CUDA_HOST_DEVICE reference back() {
      assert(_size);
      return (*this)[_size - 1];
    }

    CUDA_HOST_DEVICE const_reference back() const {
      assert(_size);
      return (*this)[_size - 1];
    }

    CUDA_HOST_DEVICE reference operator[](std::size_t i) {
      assert(i < _size);
      return *at(i);
    }

    CUDA_HOST_DEVICE const_reference operator[](std::size_t i) const {
      assert(i < _size);
      return *at(i);
    }

    /// Remove the last value from the tape.
    CUDA_HOST_DEVICE void pop_back() {
      assert(_size);
      _size--;
      at(_size)->~T();
    }

  private:
    // Returns pointer to element at specified index, handling SBO or slab lookup
    CUDA_HOST_DEVICE T* at(std::size_t index) {
      if (index < SBO_SIZE) {
        return sbo_elements() + index;
      }
      Slab* slab = m_head;
      std::size_t idx = (index - SBO_SIZE) / SLAB_SIZE;
      while (idx--) slab = slab->next;
      return slab->elements() + ((index - SBO_SIZE) % SLAB_SIZE);
      
    }

    CUDA_HOST_DEVICE const T* at(std::size_t index) const {
      if (index < SBO_SIZE) {
        return sbo_elements() + index;
      }
      Slab* slab = m_head;
      std::size_t idx = (index - SBO_SIZE) / SLAB_SIZE;
      while (idx--) slab = slab->next;
      return slab->elements() + ((index - SBO_SIZE) % SLAB_SIZE);
      
    }

    template <typename It>
    using value_type_of = decltype(*std::declval<It>());

    // Call destructor for every value in the given range.
    template <typename It>
    static typename std::enable_if<
        !std::is_trivially_destructible<value_type_of<It>>::value>::type
    destroy(It B, It E) {
      for (It I = E - 1; I >= B; --I)
        I->~value_type_of<It>();
    }

    // If type is trivially destructible, its destructor is no-op, so we can avoid for loop here.
    template <typename It>
    static typename std::enable_if<
        std::is_trivially_destructible<value_type_of<It>>::value>::type
    CUDA_HOST_DEVICE destroy(It B, It E) {}

    /// Destroys all elements and deallocates slabs
    void clear() {
      std::size_t count = _size;

      for (std::size_t i = 0; i < SBO_SIZE && count > 0; ++i, --count) {
        sbo_elements()[i].~T();
      }

      Slab* slab = m_head;
      while (slab) {
        T* elems = slab->elements();
        for (size_t i = 0; i < SLAB_SIZE && count > 0; ++i, --count) {
          (elems + i)->~T();
        }
        Slab* tmp = slab;
        slab = slab->next;
        delete tmp;
      }

      m_head = nullptr;
      _size = 0;
      m_using_sbo = true;
    }
  };
}

#endif // CLAD_TAPE_H
