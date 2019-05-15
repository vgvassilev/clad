#ifndef CLAD_TAPE_H
#define CLAD_TAPE_H

#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>

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
   
    /// Add new value of type T constructed from args to the end of the tape.
    template <typename... ArgsT>
    void emplace_back(ArgsT&&... args) {
      if (_size >= _capacity)
        grow();
      ::new (end()) T(std::forward<ArgsT>(args)...);
      _size += 1;
    }
   
    std::size_t size() const {
      return size;
    }
   
    iterator begin() {
      return reinterpret_cast<iterator>(_data);
    }
    const_iterator begin() const {
      return reinterpret_cast<const_iterator>(_data);
    }
    iterator end() {
      return reinterpret_cast<iterator>(_data) + _size;
    }
    const_iterator end() const {
      return reinterpret_cast<const_iterator>(_data) + _size;
    }

    /// Access last value (must not be empty).
    reference back() {
      assert(_size);
      return begin()[_size - 1];
    }
    const_reference back() const {
      assert(_size);
      return begin()[_size - 1];
    }

    /// Remove the last value from the tape.
    void pop_back() {
      assert(_size);
      _size -= 1;
      end()->~T();
    }

  private:
    /// Initial capacity (allocated whenever a value is pushed into empty tape).
    constexpr static std::size_t _init_capacity = 32;
    void grow() {
      // If empty, use initial capacity.
      if (!_capacity)
        _capacity = _init_capacity;
      else
         // Double the capacity on each reallocation.
        _capacity *= 2;
      // Allocate raw storage (without calling constructors of T) of new capacity.
      T* new_data = static_cast<T*>(::operator new(_capacity * sizeof(T),
                                                   std::nothrow));
      assert(new_data);
      // Move values from old storage to the new storage. Should call move
      // constructors on non-trivial types, otherwise is expected to use
      // memcpy/memmove.
      std::uninitialized_copy(std::make_move_iterator(begin()),
                              std::make_move_iterator(end()), new_data);
      // Destroy all values in the old storage.
      destroy(begin(), end());
      _data = new_data;
    }
    
    template <typename It>
    using value_type_of = decltype(*std::declval<It>());

    // Call destructor for every value in the given range.
    template <typename It>
    static typename std::enable_if<!std::is_trivially_destructible<value_type_of<It>>::value>::type
    destroy(It B, It E) {
      for (It I = E - 1; I >= B; --I)
        I->~value_type_of<It>();
    }
    // If type is trivially destructible, its destructor is no-op, so we can avoid
    // for loop here.
    template <typename It>
    static typename std::enable_if<std::is_trivially_destructible<value_type_of<It>>::value>::type
    destroy(It B, It E) {}
  };
}

#endif // CLAD_TAPE_H
