#ifndef CLAD_DIFFERENTIATOR_SPARSITYPATTERN_H
#define CLAD_DIFFERENTIATOR_SPARSITYPATTERN_H

#include <assert.h>

#include <cstddef>
#include <initializer_list>
#include <type_traits>

namespace clad {
template <typename T> class sparsity_pattern {
  int* m_col = nullptr;
  int* m_row = nullptr;
  T* m_vals = nullptr;

  std::size_t m_col_size;
  std::size_t m_row_size;

public:
  sparsity_pattern() = default;

  void set_col_idx(std::initializer_list<int> col) {
    m_col_size = col.size();
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    m_col = new int[m_col_size];
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    m_vals = new T[m_col_size];

    std::size_t i = 0;
    for (const auto& e : col)
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      m_col[i++] = e;
  }

  void set_row_idx(std::initializer_list<int> row) {
    m_row_size = row.size();
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    m_row = new int[m_row_size];
    std::size_t i = 0;
    for (const auto& e : row)
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      m_row[i++] = e;
  }

  T& operator[](int i) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return m_vals[i];
  }

  int get_col(int idx) {
    assert(idx < m_col_size);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return m_col[idx];
  }

  int get_row(int idx) {
    assert(idx < m_row_size);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return m_row[idx];
  }
};
} // namespace clad

#endif // CLAD_DIFFERENTIATOR_SPARSITYPATTERN_H
