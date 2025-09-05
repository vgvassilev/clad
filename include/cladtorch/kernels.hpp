#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
// #define OMP _OPENMP
#ifdef OMP
#include <omp.h>
#endif

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#define CLAD_ASSERT(condition, message) assert((condition) && message)

// -------------------- Kernel Functions (Operating on raw pointers) --------------------
// These functions are low-level, high-performance routines that operate on raw C-style
// arrays. Their performance and interface are independent of how the Tensor class manages its data.
namespace cladtorch::kernels {
inline float gelu_kernel(float x) {
  constexpr float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2 / pi)
  return 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

inline void softmax_kernel(const float* logits, float* probs, int size, int end, int vocab_size) {
  CLAD_ASSERT(size > 0, "Softmax kernel requires size > 0");
  CLAD_ASSERT(end > 0, "Softmax kernel requires end > 0");
  CLAD_ASSERT(end <= size, "End index cannot exceed size");

  int V = vocab_size > 0 ? vocab_size : size;

  // Find maximum value for numerical stability
  float maxv = -10000.0f;
  for (size_t j = 0; j < end; j++)
    maxv = fmaxf(maxv, logits[j]);

  // Compute exponentials and sum
  float sum = 0.0f;
  for (size_t j = 0; j < end; j++) {
    probs[j] = expf(logits[j] - maxv);
    sum += probs[j];
  }

  // Normalize probabilities (handle division by zero)
  if (sum > 0.0f) {
    float inv_sum = 1.0f / sum;
    for (size_t j = 0; j < (size_t)end; j++)
      probs[j] = probs[j] * inv_sum;
  } else {
    // If sum is zero, set uniform distribution over valid range
    float uniform_prob = 1.0f / end;
    for (size_t j = 0; j < (size_t)end; j++)
      probs[j] = uniform_prob;
  }

  // [end, V) is padded with 0.0f due to the causal mask
  // [V, m) is padded with 0.0f due to the padded vocab
  for (size_t j = end; j < size; j++)
    probs[j] = 0.0f;
}

inline float cross_entropy_loss_kernel(const float* probs, int target_class, int size) {
  if (target_class < 0 || target_class >= size)
    return -std::log(1e-9f);
  float prob_at_target = probs[target_class];
  return -std::log(std::max(prob_at_target, 1e-9f));
}

inline void mat_vec_mul_kernel(const float* mat, const float* vec, float* result, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    result[i] = 0.0f;
    for (int j = 0; j < cols; ++j)
      result[i] += mat[i * cols + j] * vec[j];
  }
}

inline void
mat_mul_kernel_naive(const float* a_data, const float* b_data, float* result_data, size_t R, size_t C1, size_t C2) {
#pragma omp parallel for
  for (size_t i = 0; i < R; ++i) {
    for (size_t j = 0; j < C2; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < C1; ++k)
        sum += a_data[i * C1 + k] * b_data[k * C2 + j];
      result_data[i * C2 + j] = sum;
    }
  }
}

static constexpr int UNROLL = 8;
inline void mat_mul_kernel_unrolled(const float* a, const float* b, float* out, size_t R, size_t C1, size_t C2) {
  // we assume R % UNROLL == 0 (fall back otherwise)
  size_t RT = R; // R = B*T for us
#pragma omp parallel for
  for (size_t r0 = 0; r0 < RT; r0 += UNROLL) {
    for (size_t j = 0; j < C2; ++j) {
      // roll UNROLL outputs into registers
      float regs[UNROLL];
      for (int u = 0; u < UNROLL; ++u)
        regs[u] = 0.0f;
      // inner "k" loop: load one weight and multiply–accumulate into all regs
      const float* brow = b + j;
      for (size_t k = 0; k < C1; ++k) {
        float w = *(brow + k * C2);
        const float* arow = a + (r0 + 0) * C1 + k;
        for (int u = 0; u < UNROLL; ++u) {
          regs[u] += *arow * w;
          arow += C1;
        }
      }
      // write back
      for (int u = 0; u < UNROLL; ++u)
        out[(r0 + u) * C2 + j] = regs[u];
    }
  }
}

inline void mat_mul_kernel(const float* a, const float* b, float* out, size_t R, size_t C1, size_t C2) {
// Dispatch to unrolled or regular kernel based on R
#ifdef __APPLE__
  // Use Accelerate framework for unrolled matrix multiplication
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, R, C2, C1, 1.0f, a, C1, b, C2, 0.0f, out, C2);
  return;
#endif
  if (R % UNROLL == 0)
    mat_mul_kernel_unrolled(a, b, out, R, C1, C2);
  else
    mat_mul_kernel_naive(a, b, out, R, C1, C2);
}

inline void batched_mat_mul_kernel(
  const float* a_data, const float* b_data, float* result_data, size_t batch_size, size_t R, size_t C1, size_t C2
) {
  size_t a_batch_stride = R * C1;
  size_t b_batch_stride = C1 * C2;
  size_t result_batch_stride = R * C2;

  for (size_t batch = 0; batch < batch_size; ++batch) {
    const float* a_batch = a_data + batch * a_batch_stride;
    const float* b_batch = b_data + batch * b_batch_stride;
    float* result_batch = result_data + batch * result_batch_stride;
    mat_mul_kernel(a_batch, b_batch, result_batch, R, C1, C2);
  }
}

// -------------------- Linear Layer Kernels (Fused Matrix Multiplication + Bias) --------------------

inline void linear_kernel_naive(
  const float* input, const float* weight, const float* bias, float* output, size_t batch_seq, size_t in_features,
  size_t out_features
) {
  // input: [batch_seq, in_features]
  // weight: [out_features, in_features]
  // bias: [out_features]
  // output: [batch_seq, out_features]
  // Computes: output = input @ weight.T + bias

#pragma omp parallel for
  for (size_t i = 0; i < batch_seq; ++i) {
    for (size_t j = 0; j < out_features; ++j) {
      float sum = bias[j]; // Start with bias
      for (size_t k = 0; k < in_features; ++k)
        sum += input[i * in_features + k] * weight[j * in_features + k];
      output[i * out_features + j] = sum;
    }
  }
}
template <typename T>
inline void linear_kernel_unrolled(
  const T* input, const T* weight, const T* bias, T* output, size_t batch_seq, size_t in_features, size_t out_features
) {
// Unrolled version for better performance when batch_seq % UNROLL == 0
#pragma omp parallel for
  for (size_t i0 = 0; i0 < batch_seq; i0 += UNROLL) {
    for (size_t j = 0; j < out_features; ++j) {
      // Initialize registers with bias
      float regs[UNROLL];
      for (int u = 0; u < UNROLL; ++u)
        regs[u] = bias[j];

      // Accumulate input * weight for each output feature
      for (size_t k = 0; k < in_features; ++k) {
        float w = weight[j * in_features + k];
        for (int u = 0; u < UNROLL; ++u)
          regs[u] += input[(i0 + u) * in_features + k] * w;
      }

      // Write back results
      for (int u = 0; u < UNROLL; ++u)
        output[(i0 + u) * out_features + j] = regs[u];
    }
  }
}

inline void linear_kernel(
  const float* input, const float* weight, const float* bias, float* output, size_t batch_seq, size_t in_features,
  size_t out_features
) {
#ifdef __APPLE__
// 1) Broadcast the bias into each row of `output`
//    output[i, j] = bias[j]
#pragma omp parallel for
  for (size_t i = 0; i < batch_seq; ++i) {
    float* out_row = output + i * out_features;
    for (size_t j = 0; j < out_features; ++j)
      out_row[j] = bias[j];
  }
  // 2) Compute output += input @ weight^T
  //
  // In BLAS terms (row-major):
  //   C := α·A·Bᵀ + β·C
  //
  // A = input        (batch_seq × in_features)
  // B = weight       (out_features × in_features)
  // Bᵀ = weightᵀ    (in_features × out_features)
  // C = output       (batch_seq × out_features)
  //
  // We want α=1.0, β=1.0 (so that C starts at bias and adds the matmul).
  cblas_sgemm(
    /* order     */ CblasRowMajor,
    /* transA    */ CblasNoTrans,
    /* transB    */ CblasTrans,
    /* M,N,K     */ (int)batch_seq, (int)out_features, (int)in_features,
    /* α         */ 1.0f,
    /* A, lda    */ input, (int)in_features,
    /* B, ldb    */ weight, (int)in_features,
    /* β, C, ldc */ 1.0f, output, (int)out_features
  );
  return;
#endif
  // Dispatch to unrolled or regular kernel based on batch_seq
  if (batch_seq % UNROLL == 0 && batch_seq >= UNROLL)
    linear_kernel_unrolled(input, weight, bias, output, batch_seq, in_features, out_features);
  else
    linear_kernel_naive(input, weight, bias, output, batch_seq, in_features, out_features);
}
template <typename T> inline void element_wise_add_kernel(const T* a, const T* b, T* r, size_t n) {
  for (size_t i = 0; i < n; ++i)
    r[i] = a[i] + b[i];
}
template <typename T> inline void element_wise_sub_kernel(const T* a, const T* b, T* r, size_t n) {
  for (size_t i = 0; i < n; ++i)
    r[i] = a[i] - b[i];
}
template <typename T> inline void element_wise_mul_kernel(const T* a, const T* b, T* r, size_t n) {
  for (size_t i = 0; i < n; ++i)
    r[i] = a[i] * b[i];
}
template <typename T> inline void element_wise_div_kernel(const T* a, const T* b, T* r, size_t n) {
  for (size_t i = 0; i < n; ++i)
    r[i] = a[i] / b[i];
}
template <typename T> inline void scalar_add_kernel(const T* in, T s, T* r, size_t n) {
  for (size_t i = 0; i < n; ++i)
    r[i] = in[i] + s;
}
template <typename T> inline void scalar_mul_kernel(const T* in, T s, T* r, size_t n) {
  for (size_t i = 0; i < n; ++i)
    r[i] = in[i] * s;
}
template <typename T> inline void scalar_div_kernel(const T* in, T s, T* r, size_t n) {
  CLAD_ASSERT(s != 0.0f, "Division by zero.");
  for (size_t i = 0; i < n; ++i)
    r[i] = in[i] / s;
}

template <typename T>
inline void lookup_kernel(
  const T* src_data, const int* indices, T* dst_data, size_t num_indices, size_t src_first_dim, size_t slice_size
) {
  for (size_t i = 0; i < num_indices; ++i) {
    int idx = indices[i];
    CLAD_ASSERT(idx >= 0 && idx < (int)src_first_dim, "Index out of bounds in lookup.");

    const T* src_slice = src_data + idx * slice_size;
    T* dst_slice = dst_data + i * slice_size;

    for (size_t j = 0; j < slice_size; ++j)
      dst_slice[j] = src_slice[j];
  }
}

template <typename T>
inline void view_kernel(
  const T* src_data, T* dst_data, const std::vector<int>& src_shape, const std::vector<int>& src_strides,
  const std::vector<int>& dst_shape, const std::vector<int>& dst_strides, size_t split_axis, size_t offset
) {
  size_t dst_elements = 1;
  for (size_t dim : dst_shape)
    dst_elements *= dim;

  // Fast path: contiguous memory copy when possible
  if (dst_strides.size() > 0 && dst_strides.back() == 1 && src_strides.back() == 1 &&
      split_axis == dst_shape.size() - 1 && split_axis < src_strides.size() && 
      offset < static_cast<size_t>(src_shape[split_axis])) {
    size_t copy_size = dst_shape[split_axis];
    size_t outer_elements = dst_elements / copy_size;
    size_t src_stride = src_strides[split_axis];
    
    for (size_t i = 0; i < outer_elements; ++i) {
      const T* src_ptr = src_data + offset * src_stride + i * src_stride * src_shape[split_axis];
      T* dst_ptr = dst_data + i * copy_size;
      std::memcpy(dst_ptr, src_ptr, copy_size * sizeof(T));
    }
    return;
  }

  // Optimized path: direct stride calculation with bounds checking
  if (split_axis >= src_strides.size()) return;
  
  size_t split_stride = src_strides[split_axis];
  size_t base_offset = offset * split_stride;
  
  // Calculate strides for coordinate-free indexing
  std::vector<size_t> coord_multipliers(dst_shape.size());
  if (dst_shape.size() > 0) {
    coord_multipliers.back() = 1;
    for (long i = static_cast<long>(dst_shape.size()) - 2; i >= 0; --i) {
      coord_multipliers[i] = coord_multipliers[i + 1] * dst_shape[i + 1];
    }
  }

#ifdef OMP
  #pragma omp parallel for if(dst_elements > 10000)
#endif
  for (size_t dst_idx = 0; dst_idx < dst_elements; ++dst_idx) {
    size_t src_idx = base_offset;
    size_t temp_idx = dst_idx;
    
    // Direct calculation without coordinate arrays
    for (size_t i = 0; i < dst_shape.size() && i < src_strides.size(); ++i) {
      if (coord_multipliers[i] > 0) {
        size_t coord = temp_idx / coord_multipliers[i];
        temp_idx %= coord_multipliers[i];
        if (i == split_axis) {
          coord += offset;
        }
        src_idx += coord * src_strides[i];
      }
    }
    
    dst_data[dst_idx] = src_data[src_idx];
  }
}

#if defined(__SSE2__) || defined(__AVX__)
#include <immintrin.h>
#endif

// Optimized 2D matrix transpose for common case
template <typename T> inline void transpose_2d_kernel(const T* src_data, T* dst_data, size_t rows, size_t cols) {
  constexpr size_t BLOCK_SIZE = 64; // Cache-friendly block size

  for (size_t i = 0; i < rows; i += BLOCK_SIZE) {
    for (size_t j = 0; j < cols; j += BLOCK_SIZE) {
      size_t max_i = std::min(i + BLOCK_SIZE, rows);
      size_t max_j = std::min(j + BLOCK_SIZE, cols);

      for (size_t ii = i; ii < max_i; ++ii)
        for (size_t jj = j; jj < max_j; ++jj)
          dst_data[jj * rows + ii] = src_data[ii * cols + jj];
    }
  }
}

// SIMD-optimized transpose for float/double types
template <> inline void transpose_2d_kernel<float>(const float* src_data, float* dst_data, size_t rows, size_t cols) {
  constexpr size_t BLOCK_SIZE = 64;

#ifdef OMP
#pragma omp parallel for collapse(2) if (rows * cols > 10000)
#endif
  for (size_t i = 0; i < rows; i += BLOCK_SIZE) {
    for (size_t j = 0; j < cols; j += BLOCK_SIZE) {
      size_t max_i = std::min(i + BLOCK_SIZE, rows);
      size_t max_j = std::min(j + BLOCK_SIZE, cols);

#ifdef __AVX__
      // AVX vectorized inner loop for aligned blocks with bounds checking
      if ((max_j - j) >= 8 && (max_i - i) >= 8) {
        for (size_t ii = i; ii < max_i && ii + 7 < rows; ii += 8) {
          for (size_t jj = j; jj < max_j && jj + 7 < cols; jj += 8) {
            // Load 8x8 block and transpose using AVX with bounds checking
            __m256 row[8];
            for (int k = 0; k < 8; ++k) {
              if (ii + k < rows && jj + 7 < cols) {
                row[k] = _mm256_loadu_ps(&src_data[(ii + k) * cols + jj]);
              }
            }
            
            // Transpose 8x8 block
            __m256 tmp[8];
            for (int k = 0; k < 4; ++k) {
              tmp[k] = _mm256_unpacklo_ps(row[2*k], row[2*k+1]);
              tmp[4+k] = _mm256_unpackhi_ps(row[2*k], row[2*k+1]);
            }
            
            for (int k = 0; k < 8; ++k) {
              if (jj + k < cols && ii + 7 < rows) {
                _mm256_storeu_ps(&dst_data[(jj + k) * rows + ii], tmp[k]);
              }
            }
          }
        }
      } else
#endif
      {
        // Fallback scalar version
        for (size_t ii = i; ii < max_i; ++ii)
          for (size_t jj = j; jj < max_j; ++jj)
            dst_data[jj * rows + ii] = src_data[ii * cols + jj];
      }
    }
  }
}
// Cache-efficient transpose with prefetching
template <typename T>
inline void transpose_cache_optimized(
  const T* src_data, T* dst_data, size_t rows, size_t cols,
  size_t src_row_stride, size_t dst_row_stride
) {
  constexpr size_t CACHE_LINE = 64 / sizeof(T);
  constexpr size_t BLOCK_SIZE = std::max(size_t(32), CACHE_LINE * 4);
  
#ifdef OMP
  #pragma omp parallel for collapse(2) if(rows * cols > 50000)
#endif
  for (size_t i = 0; i < rows; i += BLOCK_SIZE) {
    for (size_t j = 0; j < cols; j += BLOCK_SIZE) {
      size_t max_i = std::min(i + BLOCK_SIZE, rows);
      size_t max_j = std::min(j + BLOCK_SIZE, cols);
      
      for (size_t ii = i; ii < max_i; ++ii) {
        for (size_t jj = j; jj < max_j; ++jj) {
          if (ii < rows && jj < cols && jj < src_row_stride && ii < dst_row_stride) {
            dst_data[jj * dst_row_stride + ii] = src_data[ii * src_row_stride + jj];
          }
        }
      }
    }
  }
}

// Direct stride calculation for general transpose
template <typename T>
inline void transpose_direct_kernel(
  const T* src_data, T* dst_data, const std::vector<int>& src_shape, const std::vector<int>& src_strides,
  const std::vector<int>& dst_strides, size_t dim0, size_t dim1
) {
  // Calculate stride differences for the transposed dimensions
  int src_stride0 = src_strides[dim0];
  int src_stride1 = src_strides[dim1];
  int dst_stride0 = dst_strides[dim0];
  int dst_stride1 = dst_strides[dim1];

  // Calculate total elements and outer loop structure
  size_t total_elements = 1;
  for (size_t dim : src_shape)
    total_elements *= dim;

  size_t dim0_size = src_shape[dim0];
  size_t dim1_size = src_shape[dim1];
  size_t inner_elements = dim0_size * dim1_size;
  size_t outer_elements = total_elements / inner_elements;

  // Check if we can use cache-optimized version for contiguous 2D slices
  bool is_contiguous_2d =
    (src_stride1 == 1 && dst_stride0 == 1 && src_stride0 == static_cast<int>(dim1_size) &&
     dst_stride1 == static_cast<int>(dim0_size));

  if (is_contiguous_2d && outer_elements == 1) {
    transpose_cache_optimized(src_data, dst_data, dim0_size, dim1_size, src_stride0, dst_stride1);
    return;
  }

#ifdef OMP
#pragma omp parallel for if (outer_elements > 100)
#endif
  for (size_t outer = 0; outer < outer_elements; ++outer) {
    // Calculate base indices for this outer iteration with bounds checking
    size_t outer_src_base = 0;
    size_t outer_dst_base = 0;
    size_t temp_outer = outer;
    
    for (size_t d = 0; d < src_shape.size() && d < src_strides.size() && d < dst_strides.size(); ++d) {
      if (d == dim0 || d == dim1) continue;
      
      if (src_shape[d] > 0) {
        size_t coord = temp_outer % src_shape[d];
        temp_outer /= src_shape[d];
        
        outer_src_base += coord * src_strides[d];
        outer_dst_base += coord * dst_strides[d];
      }
    }

    // Transpose the 2D slice with blocking for better cache usage
    constexpr size_t BLOCK = 32;
    for (size_t i = 0; i < dim0_size; i += BLOCK) {
      for (size_t j = 0; j < dim1_size; j += BLOCK) {
        size_t max_i = std::min(i + BLOCK, dim0_size);
        size_t max_j = std::min(j + BLOCK, dim1_size);

        for (size_t ii = i; ii < max_i; ++ii) {
          for (size_t jj = j; jj < max_j; ++jj) {
            size_t src_idx = outer_src_base + ii * src_stride0 + jj * src_stride1;
            size_t dst_idx = outer_dst_base + jj * dst_stride0 + ii * dst_stride1;
            dst_data[dst_idx] = src_data[src_idx];
          }
        }
      }
    }
  }
}

// Performance utilities
namespace transpose_perf {
inline size_t calculate_memory_bandwidth(size_t elements, size_t element_size) {
  return elements * element_size * 2; // read + write
}

inline bool should_use_parallel(size_t total_elements) {
  return total_elements > 50000; // Threshold for parallel processing
}

inline bool is_cache_friendly_size(size_t dim0, size_t dim1) {
  return (dim0 * dim1 * sizeof(float)) < (1024 * 1024); // < 1MB
}
} // namespace transpose_perf

template <typename T>
inline void transpose_kernel(
  const T* src_data, T* dst_data, const std::vector<int>& src_shape, const std::vector<int>& src_strides,
  const std::vector<int>& dst_strides, size_t dim0, size_t dim1
) {
  size_t total_elements = 1;
  for (size_t dim : src_shape)
    total_elements *= dim;

  // Strategy 1: Optimized 2D transpose for contiguous matrices
  if (src_shape.size() == 2 && dim0 == 0 && dim1 == 1 && src_strides[1] == 1 && src_strides[0] == src_shape[1] &&
      dst_strides[0] == 1 && dst_strides[1] == src_shape[0]) {

    // Choose between SIMD and cache-optimized versions
    if (transpose_perf::is_cache_friendly_size(src_shape[0], src_shape[1]))
      transpose_2d_kernel(src_data, dst_data, src_shape[0], src_shape[1]);
    else
      transpose_cache_optimized(src_data, dst_data, src_shape[0], src_shape[1], src_strides[0], dst_strides[1]);
    return;
  }

  // Strategy 2: Handle large tensors with minimal dimension count
  if (src_shape.size() <= 4 && transpose_perf::should_use_parallel(total_elements)) {
    transpose_direct_kernel(src_data, dst_data, src_shape, src_strides, dst_strides, dim0, dim1);
    return;
  }

  // Strategy 3: Fallback to direct method for complex cases
  transpose_direct_kernel(src_data, dst_data, src_shape, src_strides, dst_strides, dim0, dim1);
}

inline float vec_mean_kernel(size_t vec_size, const float* src) {
  float sum = 0.0f;
  for (size_t i = 0; i < vec_size; i++)
    sum += src[i];
  return sum / vec_size;
}

inline float vec_rstd_kernel(size_t vec_size, const float* src, float mean) {
  float eps = 1e-5f;
  float sum = 0.0f;
  for (size_t i = 0; i < vec_size; i++) {
    float diff = src[i] - mean;
    sum += diff * diff;
  }
  float var = sum / vec_size;
  return 1.0f / std::sqrt(var + eps);
}

inline void norm_kernel(const float* src_data, float* dst_data, size_t num_vectors, size_t vec_size) {
  for (size_t idx = 0; idx < num_vectors; ++idx) {
    const float* vec = src_data + idx * vec_size;
    float* out = dst_data + idx * vec_size;

    // Calculate the mean and the rstd (without bias correction)
    float mean = vec_mean_kernel(vec_size, vec);
    float rstd = vec_rstd_kernel(vec_size, vec, mean);

    for (size_t i = 0; i < vec_size; i++)
      out[i] = (vec[i] - mean) * rstd;
  }
}

template <typename T>
inline void broadcast_kernel(
  const T* src_data, T* dst_data, const std::vector<int>& src_shape, const std::vector<int>& src_strides,
  const std::vector<int>& dst_shape, const std::vector<int>& dst_strides
) {
  size_t total_elements = 1;
  for (int dim : dst_shape)
    total_elements *= dim;

  // Fast path: scalar broadcasting
  if (src_shape.size() == 1 && src_shape[0] == 1) {
    T scalar_value = src_data[0];
#ifdef OMP
    #pragma omp parallel for if(total_elements > 10000)
#endif
    for (size_t i = 0; i < total_elements; ++i) {
      dst_data[i] = scalar_value;
    }
    return;
  }

  // Pre-compute effective strides for broadcasting with bounds checking
  std::vector<size_t> effective_strides(dst_shape.size(), 0);
  std::vector<size_t> coord_multipliers(dst_shape.size());
  
  if (dst_shape.size() > 0) {
    coord_multipliers.back() = 1;
    for (long i = static_cast<long>(dst_shape.size()) - 2; i >= 0; --i) {
      coord_multipliers[i] = coord_multipliers[i + 1] * dst_shape[i + 1];
    }
  }
  
  long src_dim = static_cast<long>(src_shape.size()) - 1;
  for (long dst_dim = static_cast<long>(dst_shape.size()) - 1; dst_dim >= 0 && src_dim >= 0; --dst_dim, --src_dim) {
    if (src_dim < static_cast<long>(src_shape.size()) && src_dim < static_cast<long>(src_strides.size()) &&
        dst_dim < static_cast<long>(dst_shape.size()) && src_shape[src_dim] > 1) {
      effective_strides[dst_dim] = src_strides[src_dim];
    }
    // else: remains 0 for broadcast dimensions
  }

#ifdef OMP
  #pragma omp parallel for if(total_elements > 10000)
#endif
  for (size_t dst_idx = 0; dst_idx < total_elements; ++dst_idx) {
    size_t src_idx = 0;
    size_t temp_idx = dst_idx;
    
    // Direct calculation using pre-computed strides
    for (size_t i = 0; i < dst_shape.size(); ++i) {
      size_t coord = temp_idx / coord_multipliers[i];
      temp_idx %= coord_multipliers[i];
      src_idx += coord * effective_strides[i];
    }
    
    dst_data[dst_idx] = src_data[src_idx];
  }
}

template <typename T>
inline void broadcast_add_kernel(
  const T* a_data, const T* b_data, T* result_data, const std::vector<int>& a_shape, const std::vector<int>& a_strides,
  const std::vector<int>& b_shape, const std::vector<int>& b_strides, const std::vector<int>& result_shape,
  const std::vector<int>& result_strides
) {
  size_t total_elements = 1;
  for (int dim : result_shape)
    total_elements *= dim;

  for (size_t result_idx = 0; result_idx < total_elements; ++result_idx) {
    // Convert flat index to multi-dimensional coordinates
    std::vector<int> coords(result_shape.size());
    size_t temp_idx = result_idx;
    for (long i = result_shape.size() - 1; i >= 0; --i) {
      coords[i] = temp_idx % result_shape[i];
      temp_idx /= result_shape[i];
    }

    // Get indices for a and b with broadcasting
    size_t a_idx = 0, b_idx = 0;

    // Map to a coordinates
    int a_dim = a_shape.size() - 1;
    for (int coord_dim = coords.size() - 1; coord_dim >= 0 && a_dim >= 0; --coord_dim, --a_dim) {
      int a_coord = (a_shape[a_dim] == 1) ? 0 : coords[coord_dim];
      a_idx += a_coord * a_strides[a_dim];
    }

    // Map to b coordinates
    int b_dim = b_shape.size() - 1;
    for (int coord_dim = coords.size() - 1; coord_dim >= 0 && b_dim >= 0; --coord_dim, --b_dim) {
      int b_coord = (b_shape[b_dim] == 1) ? 0 : coords[coord_dim];
      b_idx += b_coord * b_strides[b_dim];
    }

    result_data[result_idx] = a_data[a_idx] + b_data[b_idx];
  }
}

} // namespace cladtorch::kernels
