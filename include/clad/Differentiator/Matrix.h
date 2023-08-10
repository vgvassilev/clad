#ifndef CLAD_DIFFERENTIATOR_MATRIX_H
#define CLAD_DIFFERENTIATOR_MATRIX_H

#include "clad/Differentiator/Array.h"
#include "clad/Differentiator/CladConfig.h"

#include <cassert>
#include <initializer_list>
#include <type_traits>

/// Implementing a matrix class using clad::array for storing the data.
namespace clad {

/// This class is not meant to be used by the user. It is
/// used by clad internally only.
template <typename T> class matrix {
private:
  /// The data is stored in a clad::array.
  clad::array<T> m_data;
  /// The number of rows in the matrix.
  size_t m_rows;
  /// The number of columns in the matrix.
  size_t m_cols;

public:
  /// Delete the default constructor.
  matrix() = delete;

  /// Construct a matrix of size rows x cols.
  CUDA_HOST_DEVICE matrix(size_t rows, size_t cols)
      : m_data(rows * cols), m_rows(rows), m_cols(cols) {}

  /// Construct a matrix of size rows x cols and initialize it with the given
  /// value.
  template <typename U>
  CUDA_HOST_DEVICE matrix(size_t rows, size_t cols, U val)
      : m_data(rows * cols, val), m_rows(rows), m_cols(cols) {}

  /// Copy constructor.
  CUDA_HOST_DEVICE matrix(const matrix& other)
      : m_data(other.m_data), m_rows(other.m_rows), m_cols(other.m_cols) {}

  /// Move constructor.
  CUDA_HOST_DEVICE matrix(matrix&& other) noexcept
      : m_data(std::move(other.m_data)), m_rows(other.m_rows),
        m_cols(other.m_cols) {}

  /// Destructor.
  CUDA_HOST_DEVICE ~matrix() = default;

  /// Copy assignment operator.
  CUDA_HOST_DEVICE matrix& operator=(const matrix& other) {
    m_data = other.m_data;
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    return *this;
  }

  /// Move assignment operator.
  CUDA_HOST_DEVICE matrix& operator=(matrix&& other) noexcept {
    m_data = std::move(other.m_data);
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    return *this;
  }

  /// Returns the number of rows in the matrix.
  CUDA_HOST_DEVICE size_t rows() const { return m_rows; }

  /// Returns the number of columns in the matrix.
  CUDA_HOST_DEVICE size_t cols() const { return m_cols; }

  /// Returns the reference to the element at the given row and column.
  CUDA_HOST_DEVICE T& operator()(size_t row, size_t col) {
    assert(row < m_rows && col < m_cols);
    return m_data[row * m_cols + col];
  }
  CUDA_HOST_DEVICE const T& operator()(size_t row, size_t col) const {
    assert(row < m_rows && col < m_cols);
    return m_data[row * m_cols + col];
  }

  /// Returns the reference to the row at the given index.
  CUDA_HOST_DEVICE clad::array_ref<T> operator[](size_t row_idx) {
    assert(row_idx < m_rows);
    return clad::array_ref<T>(m_data).slice(row_idx * m_cols, m_cols);
  }

  /// Adding constant to matrix.
  template <typename U, typename std::enable_if<std::is_arithmetic<U>::value,
                                                int>::type = 0>
  CUDA_HOST_DEVICE matrix<T>& operator+=(U val) {
    m_data += val;
    return *this;
  }

  /// Subtracting constant from matrix.
  template <typename U, typename std::enable_if<std::is_arithmetic<U>::value,
                                                int>::type = 0>
  CUDA_HOST_DEVICE matrix<T>& operator-=(U val) {
    m_data -= val;
    return *this;
  }

  /// Multiplying matrix with constant.
  template <typename U, typename std::enable_if<std::is_arithmetic<U>::value,
                                                int>::type = 0>
  CUDA_HOST_DEVICE matrix<T>& operator*=(U val) {
    m_data *= val;
    return *this;
  }

  /// Dividing matrix by constant.
  template <typename U, typename std::enable_if<std::is_arithmetic<U>::value,
                                                int>::type = 0>
  CUDA_HOST_DEVICE matrix<T>& operator/=(U val) {
    m_data /= val;
    return *this;
  }

  /// Element wise addition of matrix with matrix.
  template <typename U>
  CUDA_HOST_DEVICE matrix<T>& operator+=(const matrix<U>& other) {
    assert(m_rows == other.m_rows && m_cols == other.m_cols);
    m_data += other.m_data;
    return *this;
  }

  /// Element wise subtraction of matrix with matrix.
  template <typename U>
  CUDA_HOST_DEVICE matrix<T>& operator-=(const matrix<U>& other) {
    assert(m_rows == other.m_rows && m_cols == other.m_cols);
    m_data -= other.m_data;
    return *this;
  }

  /// Element wise multiplication of matrix with matrix.
  template <typename U>
  CUDA_HOST_DEVICE matrix<T>& operator*=(const matrix<U>& other) {
    assert(m_rows == other.m_rows && m_cols == other.m_cols);
    m_data *= other.m_data;
    return *this;
  }

  /// Element wise division of matrix by matrix.
  template <typename U>
  CUDA_HOST_DEVICE matrix<T>& operator/=(const matrix<U>& other) {
    assert(m_rows == other.m_rows && m_cols == other.m_cols);
    m_data /= other.m_data;
    return *this;
  }

  /// Negation of matrix.
  CUDA_HOST_DEVICE matrix<T> operator-() const {
    matrix res(m_rows, m_cols);
    res.m_data = -m_data;
    return res;
  }

  /// Subtracts the number from every element in the matrix and returns the
  /// result, when the number is on the left side of the operator.
  template <typename U, typename std::enable_if<std::is_arithmetic<U>::value,
                                                int>::type = 0>
  CUDA_HOST_DEVICE friend matrix<T> operator-(U val, const matrix<T>& mat) {
    matrix res(mat.m_rows, mat.m_cols);
    res.m_data = val - mat.m_data;
    return res;
  }
}; // class matrix

// Function for creating an identity matrix of size rows x cols, with the
// diagonal offset by diag_offset.
// For example, identity_matrix(3, 3, 1) returns:
// 0 1 0
// 0 0 1
// 0 0 0
// identity_matrix(3, 3, -1) returns:
// 0 0 0
// 1 0 0
// 0 1 0
template <typename T>
CUDA_HOST_DEVICE matrix<T> identity_matrix(size_t rows, size_t cols,
                                           int diag_offset = 0) {
  matrix<T> res(rows, cols);
  for (size_t i = 0; i < rows; ++i) {
    if (i + diag_offset < cols && i + diag_offset >= 0)
      res(i, i + diag_offset) = 1;
  }
  return res;
}

/// Overloaded operator for clad::matrix which returns a new matrix.

/// Adding constant to matrix.
template <typename T, typename U,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
CUDA_HOST_DEVICE matrix<T> operator+(const matrix<T>& mat, U val) {
  matrix<T> res(mat);
  res += val;
  return res;
}

/// Adding constant to matrix, when the constant is on the left side of the
/// operator.
template <typename T, typename U,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
CUDA_HOST_DEVICE matrix<T> operator+(U val, const matrix<T>& mat) {
  return mat + val;
}

/// Multiplying matrix with constant.
template <typename T, typename U,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
CUDA_HOST_DEVICE matrix<T> operator*(const matrix<T>& mat, U val) {
  matrix<T> res(mat);
  res *= val;
  return res;
}

/// Multiplying matrix with constant, when the constant is on the left side of
/// the operator.
template <typename T, typename U,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
CUDA_HOST_DEVICE matrix<T> operator*(U val, const matrix<T>& mat) {
  return mat * val;
}

/// Dividing matrix by constant.
template <typename T, typename U,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
CUDA_HOST_DEVICE matrix<T> operator/(const matrix<T>& mat, U val) {
  matrix<T> res(mat);
  res /= val;
  return res;
}

/// Subtracting constant from matrix.
template <typename T, typename U,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
CUDA_HOST_DEVICE matrix<T> operator-(const matrix<T>& mat, U val) {
  matrix<T> res(mat);
  res -= val;
  return res;
}

/// Element wise addition of matrix with matrix.
template <typename T, typename U>
CUDA_HOST_DEVICE matrix<T> operator+(const matrix<T>& mat,
                                     const matrix<U>& other) {
  matrix<T> res(mat);
  res += other;
  return res;
}

/// Element wise subtraction of matrix with matrix.
template <typename T, typename U>
CUDA_HOST_DEVICE matrix<T> operator-(const matrix<T>& mat,
                                     const matrix<U>& other) {
  matrix<T> res(mat);
  res -= other;
  return res;
}

/// Element wise multiplication of matrix with matrix.
template <typename T, typename U>
CUDA_HOST_DEVICE matrix<T> operator*(const matrix<T>& mat,
                                     const matrix<U>& other) {
  matrix<T> res(mat);
  res *= other;
  return res;
}

} // end namespace clad

#endif // CLAD_DIFFERENTIATOR_MATRIX_H
