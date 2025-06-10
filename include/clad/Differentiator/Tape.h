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
      alignas(T) char raw_data[32 * sizeof(T)];
      Slab* next;
      Slab() : next(nullptr) {}
      T* elements() { return reinterpret_cast<T*>(raw_data); }
    };

    Slab* _head = nullptr;
    std::size_t _size = 0;

    constexpr static std::size_t slab_size = 32;

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
      if (_size % slab_size == 0) {
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
      size_t i = _size / slab_size;
      while (i--) slab = slab->next;

      ::new (slab->elements() + (_size % slab_size)) T(std::forward<ArgsT>(args)...);
      _size++;
    }

    CUDA_HOST_DEVICE std::size_t size() const { return _size; }

    CUDA_HOST_DEVICE pointer begin() {
      return _head ? _head->elements() : nullptr;
    }
    CUDA_HOST_DEVICE const_pointer begin() const {
      return _head ? _head->elements() : nullptr;
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
    CUDA_HOST_DEVICE T* at(std::size_t index) const {
      Slab* slab = _head;
      size_t i = index / slab_size;
      while (i--) slab = slab->next;
      return slab->elements() + (index % slab_size);
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
      Slab* slab = _head;
      std::size_t count = _size;
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
    }
  };
}

#endif // CLAD_TAPE_H
