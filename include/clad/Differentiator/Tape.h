#ifndef CLAD_TAPE_H
#define CLAD_TAPE_H

#include <cassert>
#include <cstdio>
#include <memory>
#include <type_traits>
#include <utility>
#include "clad/Differentiator/CladConfig.h"

namespace clad {
  /// Dynamically-sized array (std::vector-like), primarily used for storing
  /// values in reverse-mode AD inside loops.
  template <typename T>
  class tape_impl {
    T* _data = nullptr;
    std::size_t _size = 0;
    std::size_t _capacity = 0;
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

    CUDA_HOST_DEVICE ~tape_impl(){
      destroy(begin(), end());
      // delete the old data here to make sure we do not leak anything.
      ::operator delete(const_cast<void*>(
            static_cast<const volatile void*>(_data)));
    }

    /// Move values from old to new storage
    CUDA_HOST_DEVICE T* AllocateRawStorage(std::size_t _capacity) {
      #ifdef __CUDACC__
        // Allocate raw storage (without calling constructors of T) of new capacity.
        T* new_data = static_cast<T*>(::operator new(_capacity * sizeof(T)));
      #else
        T *new_data =
          static_cast<T *>(::operator new(_capacity * sizeof(T), std::nothrow));
      #endif
      return new_data;
    }

    /// Add new value of type T constructed from args to the end of the tape.
    template <typename... ArgsT>
    CUDA_HOST_DEVICE void emplace_back(ArgsT&&... args) {
      if (_size >= _capacity)
        grow();
      ::new (const_cast<void*>(static_cast<const volatile void*>(end())))
          T(std::forward<ArgsT>(args)...);
      _size += 1;
    }

    CUDA_HOST_DEVICE std::size_t size() const { return _size; }
    CUDA_HOST_DEVICE iterator begin() {
      return reinterpret_cast<iterator>(_data);
    }
    CUDA_HOST_DEVICE const_iterator begin() const {
      return reinterpret_cast<const_iterator>(_data);
    }
    CUDA_HOST_DEVICE iterator end() {
      return reinterpret_cast<iterator>(_data) + _size;
    }
    CUDA_HOST_DEVICE const_iterator end() const {
      return reinterpret_cast<const_iterator>(_data) + _size;
    }

    /// Access last value (must not be empty).
    CUDA_HOST_DEVICE reference back() {
      assert(_size);
      return begin()[_size - 1];
    }
    CUDA_HOST_DEVICE const_reference back() const {
      assert(_size);
      return begin()[_size - 1];
    }

    CUDA_HOST_DEVICE reference operator[](std::size_t i) {
      assert(i < _size);
      return begin()[i];
    }

    CUDA_HOST_DEVICE const_reference operator[](std::size_t i) const {
      assert(i < _size);
      return begin()[i];
    }

    /// Remove the last value from the tape.
    CUDA_HOST_DEVICE void pop_back() {
      assert(_size);
      _size -= 1;
      end()->~T();
    }

  private:
    // Copies the data from a storage to another.
    // Implementation taken from std::uninitialized_copy
    template <class InputIt, class NoThrowForwardIt>
    CUDA_HOST_DEVICE void MoveData(InputIt first, InputIt last,
                                   NoThrowForwardIt d_first) {
      NoThrowForwardIt current = d_first;
      // We specifically add and remove the CV qualifications here so that
      // cases where NoThrowForwardIt is CV qualified, we can still do the
      // allocation properly.
      for (; first != last; ++first, (void)++current) {
        ::new (const_cast<void*>(
            static_cast<const volatile void*>(clad_addressof(*current))))
          T(std::move(*first));
      }
    }
    /// Initial capacity (allocated whenever a value is pushed into empty tape).
    constexpr static std::size_t _init_capacity = 32;
    CUDA_HOST_DEVICE void grow() {
      // If empty, use initial capacity.
      if (!_capacity)
        _capacity = _init_capacity;
      else
        // Double the capacity on each reallocation.
        _capacity *= 2;
      T* new_data = AllocateRawStorage(_capacity);

      if (!new_data) {
        // clean up the memory mess just in case!
        destroy(begin(), end());
        printf("Allocation failure during tape resize! Aborting.\n");
        trap(EXIT_FAILURE);
      }

      // Move values from old storage to the new storage. Should call move
      // constructors on non-trivial types, otherwise is expected to use
      // memcpy/memmove.
      MoveData(begin(), end(), new_data);
      // Destroy all values in the old storage.
      destroy(begin(), end());
      // delete the old data here to make sure we do not leak anything.
      ::operator delete(const_cast<void*>(
            static_cast<const volatile void*>(_data)));
      _data = new_data;
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
    // If type is trivially destructible, its destructor is no-op, so we can avoid
    // for loop here.
    template <typename It>
    static typename std::enable_if<
        std::is_trivially_destructible<value_type_of<It>>::value>::type
    CUDA_HOST_DEVICE
    destroy(It B, It E) {}
  };
}

#endif // CLAD_TAPE_H
