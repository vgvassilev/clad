#ifndef CLAD_TAPE_H
#define CLAD_TAPE_H

#include <cassert>
#include <cstdio>
#include <memory>
#include <type_traits>
#include <utility>
#include "clad/Differentiator/CladConfig.h"

namespace clad {
  template <typename T>
  class tape_impl {
    struct Slab {
      alignas(T) char raw_data[1024 * sizeof(T)];
      Slab* next;
      Slab() : next(nullptr) {}
      T* elements() { return reinterpret_cast<T*>(raw_data); }
    };

    constexpr static std::size_t SBO_SIZE = 64;
    alignas(T) char static_buffer_[SBO_SIZE * sizeof(T)];
    bool using_sbo_ = true;

    Slab* _head = nullptr;
    std::size_t _size = 0;

    constexpr static std::size_t slab_size = 1024;

    CUDA_HOST_DEVICE T* sbo_elements() {
      return reinterpret_cast<T*>(static_buffer_);
    }

    CUDA_HOST_DEVICE const T* sbo_elements() const {
      return reinterpret_cast<const T*>(static_buffer_);
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

    template <typename... ArgsT>
    CUDA_HOST_DEVICE void emplace_back(ArgsT&&... args) {
      if (_size < SBO_SIZE) {
        ::new (const_cast<void*>(static_cast<const volatile void*>(sbo_elements() + _size)))
            T(std::forward<ArgsT>(args)...);
      } else {
        if (using_sbo_) {
          using_sbo_ = false;
        }
        if ((_size - SBO_SIZE) % slab_size == 0) {
          Slab* new_slab = new Slab();
          if (!_head) {
            _head = new_slab;
          } else {
            Slab* last = _head;
            while (last->next) last = last->next;
            last->next = new_slab;
          }
        }

        Slab* slab = _head;
        std::size_t idx = (_size - SBO_SIZE) / slab_size;
        while (idx--) slab = slab->next;

        ::new (const_cast<void*>(static_cast<const volatile void*>(
            slab->elements() + ((_size - SBO_SIZE) % slab_size))))
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

    CUDA_HOST_DEVICE void pop_back() {
      assert(_size);
      _size--;
      at(_size)->~T();
    }

  private:
    CUDA_HOST_DEVICE T* at(std::size_t index) {
      if (index < SBO_SIZE) {
        return sbo_elements() + index;
      } else {
        Slab* slab = _head;
        std::size_t idx = (index - SBO_SIZE) / slab_size;
        while (idx--) slab = slab->next;
        return slab->elements() + ((index - SBO_SIZE) % slab_size);
      }
    }

    CUDA_HOST_DEVICE const T* at(std::size_t index) const {
      if (index < SBO_SIZE) {
        return sbo_elements() + index;
      } else {
        Slab* slab = _head;
        std::size_t idx = (index - SBO_SIZE) / slab_size;
        while (idx--) slab = slab->next;
        return slab->elements() + ((index - SBO_SIZE) % slab_size);
      }
    }

    template <typename It>
    using value_type_of = decltype(*std::declval<It>());

    template <typename It>
    static typename std::enable_if<
        !std::is_trivially_destructible<value_type_of<It>>::value>::type
    destroy(It B, It E) {
      for (It I = E - 1; I >= B; --I)
        I->~value_type_of<It>();
    }

    template <typename It>
    static typename std::enable_if<
        std::is_trivially_destructible<value_type_of<It>>::value>::type
    CUDA_HOST_DEVICE destroy(It B, It E) {}

    void clear() {
      std::size_t count = _size;

      for (std::size_t i = 0; i < SBO_SIZE && count > 0; ++i, --count) {
        sbo_elements()[i].~T();
      }

      Slab* slab = _head;
      while (slab) {
        T* elems = slab->elements();
        for (size_t i = 0; i < slab_size && count > 0; ++i, --count) {
          elems[i].~T();
        }
        Slab* tmp = slab;
        slab = slab->next;
        delete tmp;
      }

      _head = nullptr;
      _size = 0;
      using_sbo_ = true;
    }
  };
}

#endif // CLAD_TAPE_H

