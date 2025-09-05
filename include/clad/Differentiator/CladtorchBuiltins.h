#include <clad/Differentiator/Array.h>
#include <clad/Differentiator/BuiltinDerivatives.h>
#include <clad/Differentiator/FunctionTraits.h>
#include <cladtorch/cladtorch.hpp>

namespace clad {
// specialize the zero_init function for Tensor
template <typename T> void zero_init(::cladtorch::Tensor<T>& tensor) {
  tensor.fill(0);
}
template <class T> void zero_init(::std::vector<::cladtorch::Tensor<T>>& p) {
  for (auto& elem : p)
    elem.fill(0);
}

namespace custom_derivatives {
namespace cladtorch {
inline ::clad::ValueAndPushforward<float, float> gelu_kernel_pushforward(float x, float _d_x) {
    const float _d_sqrt_2_over_pi = 0.F;
    const float sqrt_2_over_pi = 0.797884583F;
    float _t0 = 0.0447149985F * x;
    float _t1 = _t0 * x;
    float _t2 = (x + _t1 * x);
    ValueAndPushforward<float, float> _t3 = ::clad::custom_derivatives::std::tanh_pushforward(sqrt_2_over_pi * _t2, _d_sqrt_2_over_pi * _t2 + sqrt_2_over_pi * (_d_x + ((0.F * x + 0.0447149985F * _d_x) * x + _t0 * _d_x) * x + _t1 * _d_x));
    float _t4 = 0.5F * x;
    float _t5 = (1.F + _t3.value);
    return {_t4 * _t5, (0.F * x + 0.5F * _d_x) * _t5 + _t4 * (0.F + _t3.pushforward)};
}
void gelu_pullback(const ::cladtorch::Tensor<float> &in, ::cladtorch::Tensor<float> _d_y, ::cladtorch::Tensor<float> *_d_in) {
    for (int i=0;i<_d_y.num_elements();i++) {
        float _d_r_d0 = _d_y.data()[i];
        float _r0 = 0.F;
        _r0 += _d_r_d0 * gelu_kernel_pushforward(in.data()[i], 1.F).pushforward;
        (*_d_in).data()[i] += _r0;
    }
}
// Matrix multiplication pullback
template <typename T>
void matmul_pullback(const ::cladtorch::Tensor<T>& a, const ::cladtorch::Tensor<T>& b, ::cladtorch::Tensor<T> _d_y,
                     ::cladtorch::Tensor<T>* _d_a, ::cladtorch::Tensor<T>* _d_b) {
  // For C = matmul(A, B), the gradients are:
  // dA = matmul(dC, B^T)
  // dB = matmul(A^T, dC)

  // Handle different cases based on tensor dimensions
  if (a.ndim() == 2 && b.ndim() == 2) {
    // Case: 2D x 2D matrix multiplication
    // A: (R, C1), B: (C1, C2), C: (R, C2)
    // dA = matmul(dC, B^T) -> (R, C2) x (C2, C1) = (R, C1)
    // dB = matmul(A^T, dC) -> (C1, R) x (R, C2) = (C1, C2)
    auto b_transposed = b.transpose(0, 1);
    auto a_transposed = a.transpose(0, 1);

    auto grad_a = ::cladtorch::matmul(_d_y, b_transposed);
    auto grad_b = ::cladtorch::matmul(a_transposed, _d_y);

    *_d_a += grad_a;
    *_d_b += grad_b;
  } else if (a.ndim() == 3 && b.ndim() == 2) {
    // Case: 3D x 2D batched matrix multiplication
    // A: (B, T, C), B: (C, out_features), C: (B, T, out_features)
    // dA = matmul(dC, B^T) -> (B, T, out_features) x (out_features, C) = (B, T, C)
    // dB = matmul(A^T, dC) -> sum over batch of (C, T) x (T, out_features) = (C, out_features)

    auto b_transposed = b.transpose(0, 1);
    auto grad_a = ::cladtorch::matmul(_d_y, b_transposed);
    *_d_a += grad_a;

    // For dB, we need to sum contributions from all batch elements
    // Reshape A from (B, T, C) to (B*T, C), then transpose to (C, B*T)
    int batch_size = a.size(0), seq_len = a.size(1), channels = a.size(2);
    auto a_reshaped = a.reshape({batch_size * seq_len, channels});
    auto a_reshaped_transposed = a_reshaped.transpose(0, 1);

    // Reshape dC from (B, T, out_features) to (B*T, out_features)
    auto dy_reshaped = _d_y.reshape({batch_size * seq_len, _d_y.size(2)});

    auto grad_b = ::cladtorch::matmul(a_reshaped_transposed, dy_reshaped);
    *_d_b += grad_b;
  } else if (a.ndim() == 3 && b.ndim() == 3) {
    // Case: 3D x 3D batched matrix multiplication
    // A: (B, R, C1), B: (B, C1, C2), C: (B, R, C2)

    int B = a.size(0);
    for (int batch = 0; batch < B; ++batch) {
      // Extract batch slices - this is a simplified approach
      // In practice, you might want to implement batch-aware transpose and matmul
      // For now, we'll handle this case similarly to 2D but for each batch

      // This is a placeholder - a full implementation would need proper batch slicing
      // For now, we'll use the same logic as 2D case
      auto b_transposed = b.transpose(1, 2);
      auto a_transposed = a.transpose(1, 2);

      auto grad_a = ::cladtorch::matmul(_d_y, b_transposed);
      auto grad_b = ::cladtorch::matmul(a_transposed, _d_y);

      *_d_a += grad_a;
      *_d_b += grad_b;
    }
  } else if (a.ndim() == 4 && b.ndim() == 4) {
    // Case: 4D x 4D multi-head attention matmul
    // A: (B, H, T1, d), B: (B, H, d, T2), C: (B, H, T1, T2)

    // For 4D case, handle batch and head dimensions

    // For each batch and head, compute gradients
    // This is a simplified approach - a full implementation would be more efficient
    auto b_transposed = b.transpose(2, 3); // Transpose last two dimensions
    auto a_transposed = a.transpose(2, 3); // Transpose last two dimensions

    auto grad_a = ::cladtorch::matmul(_d_y, b_transposed);
    auto grad_b = ::cladtorch::matmul(a_transposed, _d_y);

    *_d_a += grad_a;
    *_d_b += grad_b;
  } else if (a.ndim() == 2 && b.ndim() == 1) {
    // Case: 2D x 1D matrix-vector multiplication
    // A: (R, C), B: (C,), C: (R,)
    // dA = outer_product(dC, B) -> (R,) outer (C,) = (R, C)
    // dB = matmul(A^T, dC) -> (C, R) x (R,) = (C,)

    // For dA: outer product of _d_y and b
    auto grad_a = ::cladtorch::Tensor<T>({a.size(0), a.size(1)});
    for (int r = 0; r < a.size(0); ++r)
      for (int c = 0; c < a.size(1); ++c)
        grad_a.at(r, c) = _d_y.at(r) * b.at(c);
    *_d_a += grad_a;

    // For dB: A^T * _d_y
    auto a_transposed = a.transpose(0, 1);
    auto grad_b = ::cladtorch::matmul(a_transposed, _d_y);
    *_d_b += grad_b;
  } else {
    // Unsupported case - should not happen if matmul worked
    assert(false && "Unsupported tensor dimensions for matmul pullback");
  }
}

// Softmax pullback
template <typename T>
void softmax_pullback(const ::cladtorch::Tensor<T>& input, bool is_casual, int vocab_size, ::cladtorch::Tensor<T> _d_y,
                      ::cladtorch::Tensor<T>* _d_input, bool* _d_is_casual, int* _d_vocab_size) {
  // For softmax, if y = softmax(x), then:
  // dy/dx_i = y_i * (delta_ij - y_j) where delta_ij is Kronecker delta
  // This can be written as: dy/dx = y * (grad_y - sum(grad_y * y))

  auto softmax_output = ::cladtorch::softmax(input, is_casual, vocab_size);

  // Compute sum(grad_y * y) along the last dimension
  int last_dim = input.shape().back();
  int num_vectors = input.num_elements() / last_dim;

  for (int vec = 0; vec < num_vectors; ++vec) {
    T sum_grad_y_times_y = 0;

    // Calculate the sum for this vector
    for (int i = 0; i < last_dim; ++i) {
      int idx = vec * last_dim + i;
      sum_grad_y_times_y += _d_y.data()[idx] * softmax_output.data()[idx];
    }

    // Compute gradient for each element in this vector
    for (int i = 0; i < last_dim; ++i) {
      int idx = vec * last_dim + i;
      T grad = softmax_output.data()[idx] * (_d_y.data()[idx] - sum_grad_y_times_y);
      _d_input->data()[idx] += grad;
    }
  }

  // Gradients for bool and int parameters are typically zero for softmax
  // *_d_is_casual remains unchanged (no contribution)
  // *_d_vocab_size remains unchanged (no contribution)
}

template <typename T, typename U>
void cross_entropy_loss_pullback(const ::cladtorch::Tensor<T>& probs, const ::cladtorch::Tensor<U>& targets,
                                 ::cladtorch::Tensor<T> _d_y, ::cladtorch::Tensor<T>* _d_probs,
                                 ::cladtorch::Tensor<U>* _d_targets) {
  // For cross entropy loss L = -log(p_target), the gradient is:
  // dL/dp_i = -1/p_target if i == target, 0 otherwise
  // But since we typically use softmax + cross entropy, the combined gradient is:
  // dL/dx_i = p_i - 1 if i == target, p_i otherwise
  // However, here we only have probs, so: dL/dp_i = -1/p_target if i == target

  int num_classes = probs.size(probs.ndim() - 1);
  int batch_size = probs.num_elements() / num_classes;

  // _d_y is a scalar (the loss), so we need to broadcast its gradient
  T loss_grad = _d_y.scalar();              // Extract scalar value
  T avg_loss_grad = loss_grad / batch_size; // Since we return mean loss

  for (int batch = 0; batch < batch_size; ++batch) {
    int target = targets.data()[batch];
    for (int cls = 0; cls < num_classes; ++cls) {
      int idx = batch * num_classes + cls;
      if (cls == target) {
        // Gradient is -1/p_target for the target class
        T prob_val = probs.data()[idx];
        _d_probs->data()[idx] += avg_loss_grad * (-1.0f / prob_val);
      }
      // Gradient is 0 for non-target classes (no addition needed)
    }
  }
  // Targets don't have gradients in typical scenarios
  // *_d_targets remains unchanged
}

// Cross entropy loss pullback for single instance version
template <typename T>
void cross_entropy_loss_pullback(const ::cladtorch::Tensor<T>& probs, int target_class, ::cladtorch::Tensor<T> _d_y,
                                 ::cladtorch::Tensor<T>* _d_probs, int* _d_target_class) {
  // For single instance cross entropy loss
  CLAD_ASSERT(probs.ndim() == 1, "Probs tensor must be 1D for single cross entropy loss.");
  int num_classes = probs.num_elements();

  T loss_grad = _d_y.scalar(); // Extract scalar value

  for (int cls = 0; cls < num_classes; ++cls) {
    if (cls == target_class) {
      // Gradient is -1/p_target for the target class
      T prob_val = probs.data()[cls];
      _d_probs->data()[cls] += loss_grad * (-1.0f / prob_val);
    }
    // Gradient is 0 for non-target classes (no addition needed)
  }
  // Target class doesn't have gradients in typical scenarios
  // *_d_target_class remains unchanged
}

// Linear kernel pullbacks
namespace kernels {

// Linear kernel naive pullback
void linear_kernel_naive_pullback(const float* input, const float* weight, const float* bias, float* output,
                                  size_t batch_seq, size_t in_features, size_t out_features, const float* d_output,
                                  float* d_input, float* d_weight, float* d_bias) {
  // For linear: output = input @ weight.T + bias
  // Gradients are:
  // d_input[i, k] = sum_j(d_output[i, j] * weight[j, k])
  // d_weight[j, k] = sum_i(d_output[i, j] * input[i, k])
  // d_bias[j] = sum_i(d_output[i, j])

#pragma omp parallel for
  for (size_t i = 0; i < batch_seq; ++i) {
    for (size_t k = 0; k < in_features; ++k) {
      float grad_input = 0.0f;
      for (size_t j = 0; j < out_features; ++j)
        grad_input += d_output[i * out_features + j] * weight[j * in_features + k];
      d_input[i * in_features + k] += grad_input;
    }
  }

#pragma omp parallel for
  for (size_t j = 0; j < out_features; ++j) {
    float grad_bias = 0.0f;
    for (size_t i = 0; i < batch_seq; ++i)
      grad_bias += d_output[i * out_features + j];
    d_bias[j] += grad_bias;

    for (size_t k = 0; k < in_features; ++k) {
      float grad_weight = 0.0f;
      for (size_t i = 0; i < batch_seq; ++i)
        grad_weight += d_output[i * out_features + j] * input[i * in_features + k];
      d_weight[j * in_features + k] += grad_weight;
    }
  }
}
constexpr int UNROLL = 8;

/*  Pull-back for y = x Wᵀ + b
 *
 *  input   : [batch_seq , in_features]   (row major)
 *  weight  : [out_feat  , in_features]   (row major)
 *  d_output: [batch_seq , out_features]  (row major)
 *
 *  All gradient buffers are assumed to be zero-initialised by the caller.
 *  Thread-safe: every parallel section writes to a disjoint slice.
 */
inline void linear_kernel_unrolled_pullback(const float* input, const float* weight,
                                            const float* /*bias*/, float* /*output*/, // not needed here
                                            size_t batch_seq, size_t in_features, size_t out_features,
                                            const float* d_output, float* d_input, float* d_weight, float* d_bias) {
// ---------- 1. d_input = d_output · W  -----------------------------------
#pragma omp parallel for schedule(static)
  for (size_t i0 = 0; i0 < batch_seq; i0 += UNROLL) {
    for (size_t k = 0; k < in_features; ++k) {
      float accum[UNROLL] = {0};

      for (size_t j = 0; j < out_features; ++j) {
        const float w_jk = weight[j * in_features + k]; // W[j,k]
#pragma omp simd
        for (int u = 0; u < UNROLL; ++u)
          accum[u] += d_output[(i0 + u) * out_features + j] * w_jk;
      }

#pragma omp simd
      for (int u = 0; u < UNROLL; ++u)
        d_input[(i0 + u) * in_features + k] += accum[u];
    }
  }

// ---------- 2. d_weight & d_bias  ----------------------------------------
#pragma omp parallel for schedule(static)
  for (size_t j = 0; j < out_features; ++j) {
    ::std::vector<float> local_dw(in_features, 0.0f); // private to this thread
    float local_db = 0.0f;

    for (size_t i0 = 0; i0 < batch_seq; i0 += UNROLL) {
      float dout_blk[UNROLL];

#pragma omp simd
      for (int u = 0; u < UNROLL; ++u) {
        dout_blk[u] = d_output[(i0 + u) * out_features + j];
        local_db += dout_blk[u]; // bias grad
      }

      for (size_t k = 0; k < in_features; ++k) {
        float acc = 0.0f;
#pragma omp simd reduction(+ : acc)
        for (int u = 0; u < UNROLL; ++u)
          acc += dout_blk[u] * input[(i0 + u) * in_features + k];
        local_dw[k] += acc; // weight grad
      }
    }

    // write-back – this thread is the *sole* owner of (j, :)
    d_bias[j] += local_db;
    for (size_t k = 0; k < in_features; ++k)
      d_weight[j * in_features + k] += local_dw[k];
  }
}

// Apple Accelerate optimized linear kernel pullback
inline void linear_kernel_accelerate_pullback(const float* input, const float* weight, const float* bias, float* output,
                                             size_t batch_seq, size_t in_features, size_t out_features,
                                             const float* d_output, float* d_input, float* d_weight, float* d_bias) {
#ifdef __APPLE__
  // For linear: output = input @ weight.T + bias
  // Gradients are:
  // d_input[i, k] = sum_j(d_output[i, j] * weight[j, k])  ->  d_input = d_output @ weight
  // d_weight[j, k] = sum_i(d_output[i, j] * input[i, k])  ->  d_weight = d_output.T @ input
  // d_bias[j] = sum_i(d_output[i, j])                     ->  sum over batch dimension

  // 1. Compute d_input = d_output @ weight
  // d_output: [batch_seq, out_features] (row major)
  // weight:   [out_features, in_features] (row major)
  // d_input:  [batch_seq, in_features] (row major)
  //
  // BLAS: C := α·A·B + β·C
  // A = d_output, B = weight, C = d_input
  cblas_sgemm(
    /* order     */ CblasRowMajor,
    /* transA    */ CblasNoTrans,
    /* transB    */ CblasNoTrans,
    /* M,N,K     */ (int)batch_seq, (int)in_features, (int)out_features,
    /* α         */ 1.0f,
    /* A, lda    */ d_output, (int)out_features,
    /* B, ldb    */ weight, (int)in_features,
    /* β, C, ldc */ 1.0f, d_input, (int)in_features
  );

  // 2. Compute d_weight = d_output.T @ input
  // d_output: [batch_seq, out_features] -> transpose to [out_features, batch_seq]
  // input:    [batch_seq, in_features]
  // d_weight: [out_features, in_features] (row major)
  //
  // BLAS: C := α·A^T·B + β·C
  // A = d_output (transposed), B = input, C = d_weight
  cblas_sgemm(
    /* order     */ CblasRowMajor,
    /* transA    */ CblasTrans,
    /* transB    */ CblasNoTrans,
    /* M,N,K     */ (int)out_features, (int)in_features, (int)batch_seq,
    /* α         */ 1.0f,
    /* A, lda    */ d_output, (int)out_features,
    /* B, ldb    */ input, (int)in_features,
    /* β, C, ldc */ 1.0f, d_weight, (int)in_features
  );

  // // Use cblas_sgemv to compute bias gradient efficiently
  // // d_bias = ones^T @ d_output where ones is a vector of 1s
  // // This computes the column-wise sum of d_output
  
  // // Create a temporary vector of ones for the matrix-vector multiplication
  // float* ones = (float*)malloc(batch_seq * sizeof(float));
  // for (size_t i = 0; i < batch_seq; ++i) {
  //   ones[i] = 1.0f;
  // }
  // // Compute d_bias += ones^T @ d_output using GEMV
  // // y := α·A^T·x + β·y
  // // A = d_output [batch_seq, out_features]
  // // x = ones [batch_seq]
  // // y = d_bias [out_features]
  // cblas_sgemv(
  //   /* order     */ CblasRowMajor,
  //   /* trans     */ CblasTrans,
  //   /* M, N      */ (int)batch_seq, (int)out_features,
  //   /* α         */ 1.0f,
  //   /* A, lda    */ d_output, (int)out_features,
  //   /* x, incx   */ ones, 1,
  //   /* β, y, incy */ 1.0f, d_bias, 1
  // );
  // free(ones);
  // 3. Compute d_bias = sum(d_output, dim=0)
  // x = ones [batch_seq]
  // y = d_bias [out_features]
  // Simple loop is efficient for bias computation and avoids memory allocation
  for (size_t j = 0; j < out_features; ++j) {
    float grad_bias = 0.0f;
    for (size_t i = 0; i < batch_seq; ++i) {
      grad_bias += d_output[i * out_features + j];
    }
    d_bias[j] += grad_bias;
  }
#else
  // Fallback to unrolled implementation on non-Apple platforms
  if (batch_seq % 8 == 0 && batch_seq >= 8)
    linear_kernel_unrolled_pullback(input, weight, bias, output, batch_seq, in_features, out_features, d_output,
                                    d_input, d_weight, d_bias);
  else
    linear_kernel_naive_pullback(input, weight, bias, output, batch_seq, in_features, out_features, d_output, d_input,
                                 d_weight, d_bias);
#endif
}

// Linear kernel pullback dispatcher
void linear_kernel_pullback(const float* input, const float* weight, const float* bias, float* output, size_t batch_seq,
                            size_t in_features, size_t out_features, const float* d_output, float* d_input,
                            float* d_weight, float* d_bias) {
#ifdef __APPLE__
  // Use Apple Accelerate optimized version on macOS
  linear_kernel_accelerate_pullback(input, weight, bias, output, batch_seq, in_features, out_features, d_output,
                                   d_input, d_weight, d_bias);
#else
  // Dispatch to unrolled or regular kernel based on batch_seq
  if (batch_seq % 8 == 0 && batch_seq >= 8)
    linear_kernel_unrolled_pullback(input, weight, bias, output, batch_seq, in_features, out_features, d_output,
                                    d_input, d_weight, d_bias);
  else
    linear_kernel_naive_pullback(input, weight, bias, output, batch_seq, in_features, out_features, d_output, d_input,
                                 d_weight, d_bias);
#endif
}

} // namespace kernels

// Linear function pullback
template <typename T>
void linear_pullback(const ::cladtorch::Tensor<T>& input, const ::cladtorch::Tensor<T>& weight,
                     const ::cladtorch::Tensor<T>& bias, ::cladtorch::Tensor<T> _d_y, ::cladtorch::Tensor<T>* _d_input,
                     ::cladtorch::Tensor<T>* _d_weight, ::cladtorch::Tensor<T>* _d_bias) {
  static_assert(::std::is_same<T, float>::value, "Linear pullback currently only supports float tensors");

  // Extract dimensions (same as forward pass)
  const auto& input_shape = input.shape();
  const auto& weight_shape = weight.shape();
  const auto& bias_shape = bias.shape();

  const int in_features = input_shape[input.ndim() - 1];
  const int out_features = weight_shape[0];
  const int batch_seq = input.num_elements() / in_features;

  // Call the kernel pullback
  kernels::linear_kernel_pullback(input.data(), weight.data(), bias.data(), nullptr, // output not needed for pullback
                                  static_cast<size_t>(batch_seq), static_cast<size_t>(in_features),
                                  static_cast<size_t>(out_features), _d_y.data(), _d_input->data(), _d_weight->data(),
                                  _d_bias->data());
}

} // namespace cladtorch
namespace class_functions {

template<typename T>
void scalar_pullback(const ::cladtorch::Tensor<T> *_this, T _d_y, ::cladtorch::Tensor<T> *_d_this) {
    _d_this->data()[0] += _d_y;
}

template<typename T>
clad::ValueAndAdjoint<T*, T*> data_reverse_forw(::cladtorch::Tensor<T> *_this, ::cladtorch::Tensor<T> *_d_this) {
    return {_this->data(), _d_this->data()};
}

template<typename T>
void data_pullback(::cladtorch::Tensor<T> *_this, ::cladtorch::Tensor<T> *_d_this) {
}

void operator_plus_equal_pullback(::cladtorch::Tensor<float>* _this, const ::cladtorch::Tensor<float>& other,
                                  ::cladtorch::Tensor<float> _d_y, ::cladtorch::Tensor<float>* _d_this,
                                  ::cladtorch::Tensor<float>* _d_other) {
  // For +=, _d_y flows to _d_this
  *_d_this += _d_y;
  *_d_other += _d_y;
}

void operator_plus_pullback(const ::cladtorch::Tensor<float>* _this, const ::cladtorch::Tensor<float>& other,
                            ::cladtorch::Tensor<float> _d_y, ::cladtorch::Tensor<float>* _d_this,
                            ::cladtorch::Tensor<float>* _d_other) {
  // For +, gradient flows to both operands
  *_d_this += _d_y;
  if (_d_other->shape() == _d_y.shape()) {
    *_d_other += _d_y;
  } else {
    // If shapes don't match, we assume _d_y is batched, and _d_other is not
    CLAD_ASSERT(_d_other->ndim() == 1, "_d_other needs to be 1D");
    CLAD_ASSERT(_d_y.size(_d_y.ndim() - 1) == _d_other->num_elements(),
                "Shape mismatch in operator+ pullback: _d_y and _d_other must have compatible shapes.");
    int batch_size = _d_y.num_elements() / _d_other->num_elements();
    int len = _d_other->num_elements();
    for (int i = 0; i < batch_size; i++)
      for (int j = 0; j < len; j++)
        _d_other->data()[j] += _d_y.data()[i * len + j];
    // *_d_other += _d_y; // Assuming _d_y can be broadcasted to both shapes
  }
}

// Subtraction operators
void operator_minus_equal_pullback(::cladtorch::Tensor<float>* _this, const ::cladtorch::Tensor<float>& other,
                                   ::cladtorch::Tensor<float> _d_y, ::cladtorch::Tensor<float>* _d_this,
                                   ::cladtorch::Tensor<float>* _d_other) {
  // For -=, _d_y flows to _d_this
  *_d_this += _d_y;
  *_d_other -= _d_y; // Negate gradient for _d_other
}

void operator_minus_pullback(const ::cladtorch::Tensor<float>* _this, const ::cladtorch::Tensor<float>& other,
                             ::cladtorch::Tensor<float> _d_y, ::cladtorch::Tensor<float>* _d_this,
                             ::cladtorch::Tensor<float>* _d_other) {
  // For -, gradient flows to first operand as-is
  *_d_this += _d_y;
  *_d_other -= _d_y; // Negate gradient for second operand
}

// Multiplication operators
void operator_star_equal_pullback(::cladtorch::Tensor<float>* _this, const ::cladtorch::Tensor<float>& other,
                                  ::cladtorch::Tensor<float> _d_y, ::cladtorch::Tensor<float>* _d_this,
                                  ::cladtorch::Tensor<float>* _d_other) {
  // For *=, d_this += d_y * other
  auto grad_this = _d_y * other;
  *_d_this += grad_this;
  assert(0 && "Not implemented yet");
  // For d_other, gradient is d_y * _this (before the operation)
  // Note: we need the original value of _this before the *= operation
  // This is a limitation - we'd need the original value stored
  // For now, assuming _this still contains the result after *=
  // auto current_this = *_this;
  // auto original_this = current_this / other;  // Reconstruct original value
  // auto grad_other = _d_y * original_this;
  // *_d_other += grad_other;
}

void operator_star_pullback(const ::cladtorch::Tensor<float>* _this, const ::cladtorch::Tensor<float>& other,
                            ::cladtorch::Tensor<float> _d_y, ::cladtorch::Tensor<float>* _d_this,
                            ::cladtorch::Tensor<float>* _d_other) {
  // For *, d_this += d_y * other
  auto grad_this = _d_y * other;
  *_d_this += grad_this;

  // For d_other, gradient is d_y * _this
  auto grad_other = _d_y * (*_this);
  if (grad_other.shape() == _d_other->shape()) {
    *_d_other += grad_other; // If shapes match, add directly
  } else {
    // If shapes don't match, we assume _d_y is batched, and _d_other is not
    CLAD_ASSERT(_d_other->ndim() == 1, "_d_other needs to be 1D");
    CLAD_ASSERT(_d_y.size(_d_y.ndim() - 1) == _d_other->num_elements(),
                "Shape mismatch in operator* pullback: _d_y and _d_other must have compatible shapes.");
    int batch_size = _d_y.num_elements() / _d_other->num_elements();
    int len = _d_other->num_elements();
    for (int i = 0; i < batch_size; i++)
      for (int j = 0; j < len; j++)
        _d_other->data()[j] += grad_other.data()[i * len + j];
  }
}

// Scalar multiplication operators
void operator_star_equal_pullback(::cladtorch::Tensor<float>* _this, float scalar, ::cladtorch::Tensor<float> _d_y,
                                  ::cladtorch::Tensor<float>* _d_this, float* _d_scalar) {
  // For tensor *= scalar
  auto grad_this = _d_y * scalar;
  *_d_this += grad_this;

  // For scalar gradient, sum all elements of (_d_y * original_this)
  auto current_this = *_this;
  auto original_this = current_this / scalar; // Reconstruct original value
  auto grad_scalar_tensor = _d_y * original_this;

  float grad_scalar_sum = 0;
  for (int i = 0; i < grad_scalar_tensor.num_elements(); ++i)
    grad_scalar_sum += grad_scalar_tensor.data()[i];
  *_d_scalar += grad_scalar_sum;
}

void operator_star_pullback(const ::cladtorch::Tensor<float>* _this, float scalar, ::cladtorch::Tensor<float> _d_y,
                            ::cladtorch::Tensor<float>* _d_this, float* _d_scalar) {
  // For tensor * scalar
  auto grad_this = _d_y * scalar;
  *_d_this += grad_this;

  // For scalar gradient, sum all elements of (_d_y * _this)
  auto grad_scalar_tensor = _d_y * (*_this);

  float grad_scalar_sum = 0;
  for (int i = 0; i < grad_scalar_tensor.num_elements(); ++i)
    grad_scalar_sum += grad_scalar_tensor.data()[i];
  *_d_scalar += grad_scalar_sum;
}

// Division operators
void operator_divide_equal_pullback(::cladtorch::Tensor<float>* _this, float scalar, ::cladtorch::Tensor<float> _d_y,
                                    ::cladtorch::Tensor<float>* _d_this, float* _d_scalar) {
  // For tensor /= scalar
  auto grad_this = _d_y / scalar;
  *_d_this += grad_this;

  // For scalar gradient: d_scalar = -sum(_d_y * original_this) / (scalar^2)
  auto current_this = *_this;
  auto original_this = current_this * scalar; // Reconstruct original value
  auto grad_scalar_tensor = _d_y * original_this;

  float grad_scalar_sum = 0;
  for (int i = 0; i < grad_scalar_tensor.num_elements(); ++i)
    grad_scalar_sum += grad_scalar_tensor.data()[i];
  *_d_scalar += -grad_scalar_sum / (scalar * scalar);
}

void operator_divide_pullback(const ::cladtorch::Tensor<float>* _this, float scalar, ::cladtorch::Tensor<float> _d_y,
                              ::cladtorch::Tensor<float>* _d_this, float* _d_scalar) {
  // For tensor / scalar
  auto grad_this = _d_y / scalar;
  *_d_this += grad_this;

  // For scalar gradient: d_scalar = -sum(_d_y * _this) / (scalar^2)
  auto grad_scalar_tensor = _d_y * (*_this);

  float grad_scalar_sum = 0;
  for (int i = 0; i < grad_scalar_tensor.num_elements(); ++i)
    grad_scalar_sum += grad_scalar_tensor.data()[i];
  *_d_scalar += -grad_scalar_sum / (scalar * scalar);
}

template <typename T>
::clad::ValueAndPushforward<::cladtorch::Tensor<T>&, ::cladtorch::Tensor<T>&>
operator_equal_pushforward(::cladtorch::Tensor<T>* _this, const ::cladtorch::Tensor<T>& other, ::cladtorch::Tensor<T>* _d_this,
                           const ::cladtorch::Tensor<T>& _d_other) {
  *_this = other;
  *_d_this = _d_other;
  return {*_this, *_d_this};
}

template <typename T>
::clad::ValueAndPushforward<::cladtorch::Tensor<T>&, ::cladtorch::Tensor<T>&>
operator_equal_reverse_forw(::cladtorch::Tensor<T>* _this, const ::cladtorch::Tensor<T>& other, ::cladtorch::Tensor<T>* _d_this,
                           const ::cladtorch::Tensor<T>& _d_other) {
  *_this = other;
  *_d_this = _d_other;
  return {*_this, *_d_this};
}

// template <typename T>
// ::clad::ValueAndPushforward<::cladtorch::Tensor<T>&, ::cladtorch::Tensor<T>&>
// operator_equal_reverse_forw(::cladtorch::Tensor<T>* _this, ::cladtorch::Tensor<T>&& other, ::cladtorch::Tensor<T>* _d_this,
//                            ::cladtorch::Tensor<T>&& _d_other) {
//   *_this = other;
//   *_d_this = _d_other;
//   return {*_this, *_d_this};
// }

// template <typename T>
// ::clad::ValueAndPushforward<::cladtorch::Tensor<T>&, ::cladtorch::Tensor<T>&>
// operator_equal_pushforward(::cladtorch::Tensor<T>* _this, ::cladtorch::Tensor<T>&& other, ::cladtorch::Tensor<T>* _d_this,
//                            ::cladtorch::Tensor<T>&& _d_other) {
//   *_this = other;
//   *_d_this = _d_other;
//   return {*_this, *_d_this};
// }

// template <typename T>
// ::clad::ValueAndAdjoint<::cladtorch::Tensor<T>&, ::cladtorch::Tensor<T>&>
// operator_equal_reverse_forw(::cladtorch::Tensor<T>* _this, const ::cladtorch::Tensor<T>& other,
//                             ::cladtorch::Tensor<T>* _d_this, const ::cladtorch::Tensor<T>& _d_other) {
//   *_this = other;
//   *_d_this = _d_other;
//   return {*_this, *_d_this};
// }

template <typename T>
void operator_equal_pullback(::cladtorch::Tensor<T>* _this, const ::cladtorch::Tensor<T>& other,
                             ::cladtorch::Tensor<T> _d_y, ::cladtorch::Tensor<T>* _d_this,
                             ::cladtorch::Tensor<T>* _d_other) {
  // For assignment, the gradient flows to both tensors
  *_d_other += *_d_this;
}

template <typename T>
void operator_equal_pullback(::cladtorch::Tensor<T>* _this, ::cladtorch::Tensor<T>&& other, ::cladtorch::Tensor<T> _d_y,
                             ::cladtorch::Tensor<T>* _d_this, ::cladtorch::Tensor<T>* _d_other) {
  // For assignment, the gradient flows to both tensors
  // *_d_this += _d_y;
  *_d_other += *_d_this;
}

template <typename T>
::clad::ValueAndPushforward<::cladtorch::Tensor<T>, ::cladtorch::Tensor<T>>
constructor_pushforward(ConstructorPushforwardTag<::cladtorch::Tensor<T>>, const ::cladtorch::Tensor<T>& p,
                        const ::cladtorch::Tensor<T>& d_p) {
  ::cladtorch::Tensor<T> v(p);
  ::cladtorch::Tensor<T> d_v(d_p);
  return {v, d_v};
}

template <typename T>
void constructor_pullback(const ::cladtorch::Tensor<T>& other, ::cladtorch::Tensor<T>* _d_this,
                          ::cladtorch::Tensor<T>* _d_other) {
  *_d_other += *_d_this;
  _d_this->fill(0);
}

template <typename T>
::clad::ValueAndPushforward<::cladtorch::Tensor<T>, ::cladtorch::Tensor<T>>
constructor_pushforward(ConstructorPushforwardTag<::cladtorch::Tensor<T>>, ::cladtorch::Tensor<T>&& p,
                        ::cladtorch::Tensor<T>&& d_p) {
  ::cladtorch::Tensor<T> v(::std::move(p));
  ::cladtorch::Tensor<T> d_v(::std::move(d_p));
  return {v, d_v};
}

template <typename T>
void constructor_pullback(::cladtorch::Tensor<T>&& other, ::cladtorch::Tensor<T>* _d_this,
                          ::cladtorch::Tensor<T>* _d_other) {
  *_d_other += *_d_this;
  _d_this->fill(0);
}

template <typename T>
::clad::ValueAndPushforward<::cladtorch::Tensor<T>, ::cladtorch::Tensor<T>>
constructor_pushforward(ConstructorPushforwardTag<::cladtorch::Tensor<T>>, const ::std::vector<int>& shape,
                        const ::std::vector<int>& d_shape) {
  ::cladtorch::Tensor<T> v(shape);
  ::cladtorch::Tensor<T> d_v(d_shape);
  return {v, d_v};
}

template <typename T>
void constructor_pullback(const ::std::vector<int>& shape, ::cladtorch::Tensor<T>* _d_this,
                          ::std::vector<int>* d_shape) {
  // *_d_other += *_d_this;
  // _d_this->fill(0);
}

template <typename T>
void transpose_pullback(const ::cladtorch::Tensor<T>* _this, int dim0, int dim1, ::cladtorch::Tensor<float> _d_y,
                        ::cladtorch::Tensor<T>* _d_this, int* _d_dim0, int* _d_dim1) {
  // The pullback of transpose is the transpose of the gradient with swapped dimensions
  // _d_this->print();
  // _d_y.print();
  *_d_this += _d_y.transpose(dim0, dim1);
  // _d_result->fill(0);
}

template <typename T, typename U>
void lookup_pullback(const ::cladtorch::Tensor<T>* _this, const ::cladtorch::Tensor<U>& indices,
                     ::cladtorch::Tensor<T> _d_y, ::cladtorch::Tensor<T>* _d_this, ::cladtorch::Tensor<U>* _d_indices) {
  // Indices don't have gradients since they are integers
  (void)_d_indices;

  // Calculate the size of each slice (everything after the first dimension)
  int slice_size = 1;
  for (int i = 1; i < _this->ndim(); ++i)
    slice_size *= _this->shape()[i];

  // Accumulate gradients back to the original tensor
  for (int i = 0; i < indices.num_elements(); ++i) {
    int idx = indices.data()[i];

    // Get pointers to the relevant slices
    const T* grad_slice = _d_y.data() + i * slice_size;
    T* orig_slice = _d_this->data() + idx * slice_size;

    // Accumulate gradients
    for (int j = 0; j < slice_size; ++j)
      orig_slice[j] += grad_slice[j];
  }
}

template <typename T, typename U>
void reshape_pullback(const ::cladtorch::Tensor<T>* _this, const ::std::vector<U>& new_shape,
                      ::cladtorch::Tensor<T> _d_y, ::cladtorch::Tensor<T>* _d_this, ::std::vector<U>* _d_new_shape) {
  // new_shape doesn't have gradients since it's a shape specification
  (void)_d_new_shape;

  // Reshape is just a reinterpretation of the same data, so we need to
  // reshape the gradient back to the original shape and accumulate
  // ::cladtorch::Tensor<T> reshaped_grad = ;
  *_d_this += _d_y.reshape(_this->shape());
}

template <typename T>
void norm_pullback(const ::cladtorch::Tensor<T>* _this, ::cladtorch::Tensor<T> _d_y, ::cladtorch::Tensor<T>* _d_this) {
  static_assert(::std::is_same<T, float>::value, "norm_pullback() is only supported for float tensors.");

  if (_this->num_elements() == 0)
    return;

  // Calculate number of vectors and vector size
  int vec_size = _this->shape().back(); // Last dimension
  int num_vectors = _this->num_elements() / vec_size;
  float eps = 1e-5f;

  for (int idx = 0; idx < num_vectors; ++idx) {
    const float* x_vec = _this->data() + idx * vec_size;
    const float* grad_out = _d_y.data() + idx * vec_size;
    float* grad_in = _d_this->data() + idx * vec_size;

    // Compute mean and rstd (same as forward pass)
    float mean = 0.0f;
    for (int i = 0; i < vec_size; i++)
      mean += x_vec[i];
    mean /= vec_size;

    float var = 0.0f;
    for (int i = 0; i < vec_size; i++) {
      float diff = x_vec[i] - mean;
      var += diff * diff;
    }
    var /= vec_size;
    float rstd = 1.0f / ::std::sqrt(var + eps);

    // Compute gradient statistics
    float grad_mean = 0.0f;
    for (int i = 0; i < vec_size; i++)
      grad_mean += grad_out[i];
    grad_mean /= vec_size;

    float grad_norm_mean = 0.0f;
    for (int i = 0; i < vec_size; i++)
      grad_norm_mean += grad_out[i] * (x_vec[i] - mean);
    grad_norm_mean /= vec_size;

    // Apply layer norm gradient formula
    for (int i = 0; i < vec_size; i++) {
      float normalized = (x_vec[i] - mean) * rstd;
      grad_in[i] += rstd * (grad_out[i] - grad_mean - normalized * grad_norm_mean * rstd);
    }
  }
}

template <typename T>
void split_pullback(const ::cladtorch::Tensor<T>* _this, int size, int axis, ::std::vector<::cladtorch::Tensor<T>> _d_y,
                    ::cladtorch::Tensor<T>* _d_this, int* _d_size, int* _d_axis) {
  // size and axis don't have gradients since they are parameters
  (void)_d_size;
  (void)_d_axis;

  // Split creates multiple tensors from one, so the pullback needs to
  // concatenate the gradients back along the split axis
  int num_splits = _this->shape()[axis] / size;

  // For each split, accumulate gradients back to the appropriate slice
  for (int i = 0; i < num_splits; ++i) {
    if (i < (int)_d_y.size()) {
      const ::cladtorch::Tensor<T>& split_grad = _d_y[i];

      // Calculate the offset for this split in the original tensor
      int split_offset = i * size;

      // Calculate slice size for elements after the split axis
      int slice_size = 1;
      for (int dim = axis + 1; dim < _this->ndim(); ++dim)
        slice_size *= _this->shape()[dim];

      // Calculate stride for the split axis
      int axis_stride = 1;
      for (int dim = axis + 1; dim < _this->ndim(); ++dim)
        axis_stride *= _this->shape()[dim];

      // Copy gradients back to the original tensor
      for (int elem = 0; elem < split_grad.num_elements(); ++elem) {
        // Calculate multi-dimensional coordinates in the split tensor
        ::std::vector<int> coords(split_grad.ndim());
        int temp_idx = elem;
        for (int dim = split_grad.ndim() - 1; dim >= 0; --dim) {
          coords[dim] = temp_idx % split_grad.shape()[dim];
          temp_idx /= split_grad.shape()[dim];
        }

        // Adjust coordinate for the split axis
        coords[axis] += split_offset;

        // Calculate flat index in the original tensor
        int orig_idx = 0;
        for (int dim = 0; dim < _this->ndim(); ++dim) {
          int stride = 1;
          for (int d = dim + 1; d < _this->ndim(); ++d)
            stride *= _this->shape()[d];
          orig_idx += coords[dim] * stride;
        }

        _d_this->data()[orig_idx] += split_grad.data()[elem];
      }
    }
  }
}
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad