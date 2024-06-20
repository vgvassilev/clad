#ifndef CLAD_DIFFERENTIATOR_NEWTAPE_H
#define CLAD_DIFFERENTIATOR_NEWTAPE_H

#include <cassert>
#include <cstdio>
#include <type_traits>
#include <utility>

#include "clad/Differentiator/CladConfig.h"

namespace clad {

static const int capacity = 32;

template <typename T> class Block {
public:
  T data[capacity];
  Block<T>* next;
  Block<T>* prev;
  using pointer = T*;
  using iterator = pointer;

  CUDA_HOST_DEVICE Block() {
  }

  CUDA_HOST_DEVICE ~Block() { destroy(block_begin(), block_end()); }

  Block(const Block& other) = delete;
  Block& operator=(const Block& other) = delete;

  Block(Block&& other) = delete;
  Block& operator=(const Block&& other) = delete;

  CUDA_HOST_DEVICE iterator block_begin() { return data; }

  CUDA_HOST_DEVICE iterator block_end() { return data + capacity; }

  template <typename It> using value_type_of = decltype(*std::declval<It>());

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
      CUDA_HOST_DEVICE
      destroy(It B, It E) {}
};

template <typename T> class new_tape_impl {
  using NonConstT = typename std::remove_cv<T>::type;

  Block<NonConstT>* m_cur_block = nullptr;
  std::size_t m_size = 0;

public:
  new_tape_impl() = default;

  ~new_tape_impl() { }

  new_tape_impl(new_tape_impl& other) = delete;
  new_tape_impl operator=(new_tape_impl& other) = delete;

  new_tape_impl(new_tape_impl&& other) = delete;
  new_tape_impl& operator=(new_tape_impl&& other) = delete;

  template <typename... ArgsT>

  CUDA_HOST_DEVICE void emplace_back(ArgsT&&... args) {
    if (!m_cur_block || m_size >= capacity) {
      Block<NonConstT>* prev_block = m_cur_block;
      m_cur_block =  static_cast<Block<NonConstT>*>(::operator new(sizeof(Block<NonConstT>)));
      if (prev_block != nullptr) {
        prev_block->next = m_cur_block;
        m_cur_block->prev = prev_block;
      }
      m_size = 0;
    }
    m_size += 1;
    ::new (const_cast<void*>(static_cast<const volatile void*>(end())))
        T(std::forward<ArgsT>(args)...);
  }

  [[nodiscard]] CUDA_HOST_DEVICE std::size_t size() const { return m_size; }

  CUDA_HOST_DEVICE T* end() { return m_cur_block->data + (m_size - 1); }

  CUDA_HOST_DEVICE T& back() {
    assert(m_size || m_cur_block->prev);
    return *end();
  }

  CUDA_HOST_DEVICE void pop_back() {
    assert(m_size || m_cur_block->prev);
    m_size -= 1;
    if (m_size == 0) {
      Block<NonConstT>* temp = m_cur_block;
      m_cur_block = m_cur_block->prev;
      // delete temp;
      m_size = capacity;
    }
  }

  void destroy() {
    while (m_cur_block != nullptr) {
      Block<NonConstT>* prev_block = m_cur_block->prev;
      delete m_cur_block;
      m_cur_block = prev_block;
    }
  }
};
} // namespace clad

#endif // CLAD_DIFFERENTIATOR_NEWTAPE_H
