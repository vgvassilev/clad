#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#ifdef OMP
#include <omp.h>
#endif

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <clad/Differentiator/Differentiator.h>

#include "dataloader.hpp"

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

void encoder_forward(float* out, const int* inp, float* wte, float* wpe, int B,
                     int T, int C) {
  // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing
  // token & position inp is (B,T) of integers, holding the token ids at each
  // (b,t) position wte is (V,C) of token embeddings, short for "weight token
  // embeddings" wpe is (maxT,C) of position embeddings, short for "weight
  // positional embedding"
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // seek to the output position in out[b,t,:]
      float* out_bt = out + b * T * C + t * C;
      // get the index of the token at inp[b, t]
      int ix = inp[b * T + t];
      // seek to the position in wte corresponding to the token
      float* wte_ix = wte + ix * C;
      // seek to the position in wpe corresponding to the position
      float* wpe_t = wpe + t * C;
      // add the two vectors and store the result in out[b,t,:]
      for (int i = 0; i < C; i++)
        out_bt[i] = wte_ix[i] + wpe_t[i];
    }
  }
}

void encoder_backward(float* dwte, float* dwpe, float* dout, const int* inp,
                      int B, int T, int C) {
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      float* dout_bt = dout + b * T * C + t * C;
      int ix = inp[b * T + t];
      float* dwte_ix = dwte + ix * C;
      float* dwpe_t = dwpe + t * C;
      for (int i = 0; i < C; i++) {
        float d = dout_bt[i];
        dwte_ix[i] += d;
        dwpe_t[i] += d;
      }
    }
  }
}

void layernorm_forward(float* out, float* mean, float* rstd, float* inp,
                       float* weight, float* bias, int B, int T, int C) {
  // reference:
  // https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html both inp
  // and out are (B,T,C) of the activations mean and rstd are (B,T) buffers, to
  // be used later in backward pass at each position (b,t) of the input, the
  // C-dimensional vector of activations gets normalized, then scaled and
  // shifted
  float eps = 1e-5f;
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // seek to the input position inp[b,t,:]
      float* x = inp + b * T * C + t * C;
      // calculate the mean
      float m = 0.0f;
      for (int i = 0; i < C; i++)
        m += x[i];
      m = m / C;
      // calculate the variance (without any bias correction)
      float v = 0.0f;
      for (int i = 0; i < C; i++) {
        float xshift = x[i] - m;
        v += xshift * xshift;
      }
      v = v / C;
      // calculate the rstd (reciprocal standard deviation)
      float s = 1.0f / std::sqrt(v + eps);
      // seek to the output position in out[b,t,:]
      float* out_bt = out + b * T * C + t * C;
      for (int i = 0; i < C; i++) {
        float n = (s * (x[i] - m));        // normalize
        float o = n * weight[i] + bias[i]; // scale and shift
        out_bt[i] = o;                     // write
      }
      // cache the mean and rstd for the backward pass later
      mean[b * T + t] = m;
      rstd[b * T + t] = s;
    }
  }
}

void layernorm_backward(float* dinp, float* dweight, float* dbias, float* dout,
                        float* inp, float* weight, float* mean, float* rstd,
                        int B, int T, int C) {
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      float* dout_bt = dout + b * T * C + t * C;
      float* inp_bt = inp + b * T * C + t * C;
      float* dinp_bt = dinp + b * T * C + t * C;
      float mean_bt = mean[b * T + t];
      float rstd_bt = rstd[b * T + t];

      // first: two reduce operations
      float dnorm_mean = 0.0f;
      float dnorm_norm_mean = 0.0f;
      for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
      }
      dnorm_mean = dnorm_mean / C;
      dnorm_norm_mean = dnorm_norm_mean / C;

      // now iterate again and accumulate all the gradients
      for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        // gradient contribution to bias
        dbias[i] += dout_bt[i];
        // gradient contribution to weight
        dweight[i] += norm_bti * dout_bt[i];
        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i;                    // term 1
        dval -= dnorm_mean;                 // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt;                    // final scale
        dinp_bt[i] += dval;
      }
    }
  }
}

void matmul_forward(float* out, const float* inp, const float* weight,
                    const float* bias, int B, int T, int C, int OC) {
  // BLAS implementation of matrix multiplication: out = inp @ weight^T + bias
  // inp is (B*T, C), weight is (OC, C), out is (B*T, OC)

  // First, perform the matrix multiplication: C = A * B^T
  // A = inp, B = weight, C = out
  // The BLAS sgemm function computes: C = alpha * op(A) * op(B) + beta * C
  // Here:
  // op(A) = inp (no transpose), op(B) = weight^T (transpose)
  // M = rows of op(A) = B * T
  // N = columns of op(B) = OC
  // K = columns of op(A) = C
  // alpha = 1.0, beta = 0.0 (to overwrite 'out' with the result)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B * T, OC, C, 1.0f, inp,
              C, weight, C, 0.0f, out, OC);

  // If bias is provided, add it to the output
  if (bias != nullptr) {
#pragma omp parallel for
    for (int bt = 0; bt < B * T; bt++)
      for (int o = 0; o < OC; o++)
        out[bt * OC + o] += bias[o];
  }
}

/* Same layout as forward:
 *
 *   inp : (B*T, C)  row-major   called A in BLAS comments
 *   weight : (OC, C) row-major   called B
 *   out  : (B*T, OC)             called C   (here we only have its grad ---
 * dout)
 *
 * Backward requirements:
 *
 *   dinp     = dout * weight      (1)   (B*T,OC) · (OC,C) → (B*T,C)
 *   dweight  = Σ_{b,t} dout[b,t,o] * inp[b,t]ᵀ   (2)   (OC,B*T) · (B*T,C) →
 * (OC,C) dbias[o] = Σ_{b,t} dout[b,t,o]               (3)
 *
 */

void matmul_backward(float* dinp, float* dweight, float* dbias,
                     const float* dout, const float* inp, const float* weight,
                     int B, int T, int C, int OC) {
  // In the forward pass, out = inp @ weight^T + bias
  // The gradients are calculated as follows:
  // 1. dinp    (dL/dinp)   = dout @ weight
  // 2. dweight (dL/dweight) = dout^T @ inp
  // 3. dbias   (dL/dbias)  = sum(dout, axis=0)
  // The original code accumulates gradients (+=), so we use beta=1.0f in BLAS
  // calls.

  // 1. Calculate gradient for input: dinp += dout @ weight
  //    - dout:   (B*T, OC) -> A
  //    - weight: (OC, C)   -> B
  //    - dinp:   (B*T, C)   -> C
  //    - op(A) is NoTrans, op(B) is NoTrans
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B * T, C, OC, 1.0f,
              dout, OC, weight, C, 1.0f, dinp, C);

  // 2. Calculate gradient for weights: dweight += dout^T @ inp
  //    - dout^T: (OC, B*T) -> op(A)
  //    - inp:    (B*T, C)  -> B
  //    - dweight: (OC, C)   -> C
  //    - op(A) is Trans, op(B) is NoTrans
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, OC, C, B * T, 1.0f, dout,
              OC, inp, C, 1.0f, dweight, C);

  // 3. Calculate gradient for bias: dbias += sum of dout rows
  if (dbias != nullptr) {
    // This is a reduction of dout over the batch dimension (B*T).
    // It can be expressed as a matrix-vector product: dbias += dout^T @
    // ones_vector. We use the BLAS sgemv routine: y = alpha*A*x + beta*y Here,
    // y=dbias, A=dout^T, x=ones_vector.

    // Create a temporary vector of ones for the reduction.
    std::vector<float> ones(B * T, 1.0f);

    // A = dout (B*T, OC)
    // We use dout^T, so trans = CblasTrans
    cblas_sgemv(CblasRowMajor, CblasTrans, B * T, OC, 1.0f, dout, OC,
                ones.data(), 1, 1.0f, dbias, 1);
  }
}

void attention_forward(float* out, float* preatt, float* att, float* inp, int B,
                       int T, int C, int NH) {
  // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
  // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
  // that holds the pre-attention and post-attention scores (used in backward)
  // output is (B, T, C)
  // attention is the only layer that mixes information across time
  // every other operation is applied at every (b,t) position independently
  // (and of course, no layer mixes information across batch)
  int C3 = C * 3;
  int hs = C / NH; // head size
  float hs_f = hs;
  float scale = 1.0f / std::sqrt(hs_f);

#pragma omp parallel for collapse(3)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      for (int h = 0; h < NH; h++) {
        float* query_t = inp + b * T * C3 + t * C3 + h * hs;
        float* preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
        float* att_bth = att + b * NH * T * T + h * T * T + t * T;

        // pass 1: calculate query dot key and maxval
        float maxval = -10000.0f; // TODO something better
        for (int t2 = 0; t2 <= t; t2++) {
          float* key_t2 =
              inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

          // (query_t) dot (key_t2)
          float val = 0.0f;
          for (int i = 0; i < hs; i++)
            val += query_t[i] * key_t2[i];
          val *= scale;
          if (val > maxval)
            maxval = val;

          preatt_bth[t2] = val;
        }

        // pass 2: calculate the exp and keep track of sum
        // maxval is being calculated and subtracted only for numerical
        // stability
        float expsum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
          float expv = expf(preatt_bth[t2] - maxval);
          expsum += expv;
          att_bth[t2] = expv;
        }
        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

        // pass 3: normalize to get the softmax
        for (int t2 = 0; t2 < T; t2++) {
          if (t2 <= t) {
            att_bth[t2] *= expsum_inv;
          } else {
            // causal attention mask. not strictly necessary to set to zero here
            // only doing this explicitly for debugging and checking to PyTorch
            att_bth[t2] = 0.0f;
          }
        }

        // pass 4: accumulate weighted values into the output of attention
        float* out_bth = out + b * T * C + t * C + h * hs;
        for (int i = 0; i < hs; i++)
          out_bth[i] = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
          float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs +
                            C * 2; // +C*2 because it's value
          float att_btht2 = att_bth[t2];
          for (int i = 0; i < hs; i++)
            out_bth[i] += att_btht2 * value_t2[i];
        }
      }
    }
  }
}

void attention_backward(float* __restrict dinp, float* __restrict dpreatt,
                        float* __restrict datt, float* __restrict dout,
                        float* __restrict inp, float* __restrict att, int B,
                        int T, int C, int NH) {
  const int C3 = 3 * C;
  const int hs = C / NH;
  const float scale = 1.0f / std::sqrt((float)hs);

  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < T; ++t) {
      for (int h = 0; h < NH; ++h) {
        // Pointers for this (b,t,h)
        float* att_bth = att + ((size_t)b * NH * T + h * T + t) *
                                   T; // length T, but only 0..t used
        float* datt_bth = datt + ((size_t)b * NH * T + h * T + t) * T;
        float* dpreatt_bt = dpreatt + ((size_t)b * NH * T + h * T + t) * T;

        float* q = inp + ((size_t)b * T + t) * C3 + h * hs;   // query_t
        float* dq = dinp + ((size_t)b * T + t) * C3 + h * hs; // dquery_t
        float* dout_h =
            dout + ((size_t)b * T + t) * C + h * hs; // dout head slice

        // Build views K_{<=t}, V_{<=t}, dK_{<=t}, dV_{<=t}
        // Each row corresponds to timestep t2 in 0..t
        auto row_ptr = [&](float* base, int t2, int offset) -> float* {
          return base + ((size_t)b * T + t2) * C3 + h * hs + offset;
        };
        float* K0 = inp; // +C offset for keys
        float* V0 = inp; // +2C offset for values
        float* dK0 = dinp;
        float* dV0 = dinp;

        const int M = t + 1; // number of rows (timesteps included)
        // Row-major: matrices are M x hs with leading dimension hs.

        // A) Through value accumulation: y = A V
        // dy/dV += A^T dout_h  → dV_{i,:} += a[i] * dout_h
        // dy/dA = V dout_h     → vector length M
        // 1) da_val = V_{<=t} * dout_h   (M×hs)·(hs) -> (M)
        //    GEMV row-major: m=M, n=hs
        {
          cblas_sgemv(CblasRowMajor, CblasNoTrans, M, hs, 1.0f,
                      row_ptr(V0, 0, 2 * C), hs, // matrix V_{<=t}
                      dout_h, 1, 1.0f, datt_bth,
                      1); // accumulate into datt_bth[0..t]
        }

        // 2) dV += outer(a, dout_h)  (rank-1 updates for rows 0..t)
        //    Use BLAS GER: A := alpha * x * y^T + A
        //    For each row i: dV[i,:] += a[i] * dout_h
        //    That’s exactly sger with x=a (length M), y=dout_h (length hs),
        //    A=dV (M x hs)
        {
          cblas_sger(CblasRowMajor, M, hs, 1.0f, att_bth, 1, // x = a (length M)
                     dout_h, 1,                   // y = dout_h (length hs)
                     row_ptr(dV0, 0, 2 * C), hs); // A = dV matrix
        }

        // B) Softmax backward: ds = (da - dot(a,da)) ⊙ a
        // We already have da = datt_bth (accumulated above and potentially from
        // previous passes). Compute dot = <a, da>, then ds in-place into
        // dpreatt_bt[0..t].
        float dot = 0.0f;
        for (int i = 0; i < M; ++i)
          dot += att_bth[i] * datt_bth[i];
        for (int i = 0; i < M; ++i) {
          float ds = (datt_bth[i] - dot) * att_bth[i];
          dpreatt_bt[i] += ds; // accumulate
        }

        // C) s = (K q) * scale
        // dq += scale * K^T ds
        // dK += scale * ds q^T
        // 1) dq: GEMV with K^T and vector ds (length M)
        {
          // dq += scale * K_{<=t}^T * ds
          // Row-major GEMV with transposed matrix:
          //   y = alpha * A^T * x + beta*y
          // A is M x hs (rows=t2, cols=i)
          cblas_sgemv(CblasRowMajor, CblasTrans, M, hs, scale,
                      row_ptr(K0, 0, C), hs, // K_{<=t}
                      dpreatt_bt, 1,         // x = ds
                      1.0f, dq, 1);          // y = dq
        }

        // 2) dK += scale * ds q^T     (GER rank-1 update onto dK matrix M×hs)
        {
          cblas_sger(CblasRowMajor, M, hs, scale, dpreatt_bt,
                     1,                       // x = ds (length M)
                     q, 1,                    // y = q  (length hs)
                     row_ptr(dK0, 0, C), hs); // A = dK matrix
        }
      }
    }
  }
}

inline float fast_tanhf(float x) { return 2.f / (1.f + exp(-2.f * x)) - 1.f; }

void gelu_forward(float* out, const float* inp, int N) {
  const float scale = 0.7978845608028654f; // sqrt(2/pi)
  const float beta = 0.044715f;
#pragma omp simd
  for (int i = 0; i < N; i++) {
    float x = inp[i];
    float cube = beta * x * x * x;
    float tanh_arg = scale * (x + cube);
    out[i] = 0.5f * x * (1.0f + fast_tanhf(tanh_arg));
  }
}

void gelu_backward(float* dinp, const float* inp, const float* dout, int N) {
  const float s = 0.7978845608028654f; // sqrt(2/pi)
  const float b = 0.044715f;
#pragma omp simd
  for (int i = 0; i < N; ++i) {
    float x = inp[i];
    float x2 = x * x;
    float u = s * (x + b * x2 * x);
    float t = fast_tanhf(u);               // tanh(u)
    float dt = 1.0f - t * t;               // sech^2(u) but via tanh
    float du = s * (1.0f + 3.0f * b * x2); // du/dx
    float local_grad = 0.5f * (1.0f + t) + 0.5f * x * dt * du;
    dinp[i] += local_grad * dout[i];
  }
}

void residual_forward(float* out, float* inp1, float* inp2, int N) {
#pragma omp simd
  for (int i = 0; i < N; i++)
    out[i] = inp1[i] + inp2[i];
}

void residual_backward(float* dinp1, float* dinp2, float* dout, int N) {
#pragma omp simd
  for (int i = 0; i < N; i++) {
    dinp1[i] += dout[i];
    dinp2[i] += dout[i];
  }
}

void softmax_forward(float* probs, float* logits, int B, int T, int V, int Vp) {
// output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t
// position) input: logits is (B,T,Vp) of the unnormalized log probabilities Vp
// is the padded vocab size (for efficiency), V is the "real" vocab size
// example: Vp is 50304 and V is 50257
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // probs <- softmax(logits)
      float* logits_bt = logits + b * T * Vp + t * Vp;
      float* probs_bt = probs + b * T * Vp + t * Vp;

      // maxval is only calculated and subtracted for numerical stability
      float maxval = -10000.0f; // TODO something better
      for (int i = 0; i < V; i++)
        if (logits_bt[i] > maxval)
          maxval = logits_bt[i];
      float sum = 0.0f;
      for (int i = 0; i < V; i++) {
        probs_bt[i] = expf(logits_bt[i] - maxval);
        sum += probs_bt[i];
      }
      // note we only loop to V, leaving the padded dimensions
      for (int i = 0; i < V; i++)
        probs_bt[i] /= sum;
      // for extra super safety we may wish to include this too,
      // forcing the probabilities here to be zero, but it shouldn't matter
      for (int i = V; i < Vp; i++)
        probs_bt[i] = 0.0f;
    }
  }
}
void softmax_backward(float* dlogits, float* dprobs, float* probs, int B, int T,
                      int V, int Vp) {
  // backwards through softmax
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      float* dlogits_bt = dlogits + b * T * Vp + t * Vp;
      float* dprobs_bt = dprobs + b * T * Vp + t * Vp;
      float* probs_bt = probs + b * T * Vp + t * Vp;

      // sum over all outputs. s = sum(dprobs[i] * probs[i])
      float sum = 0.0f;
      for (int i = 0; i < V; i++)
        sum += dprobs_bt[i] * probs_bt[i];

      // dlogits[i] = probs[i] * (dprobs[i] - sum)
      for (int i = 0; i < V; i++)
        dlogits_bt[i] += probs_bt[i] * (dprobs_bt[i] - sum);
    }
  }
}

void crossentropy_forward(float* losses, float* probs, const int* targets,
                          int B, int T, int Vp) {
  // output: losses is (B,T) of the individual losses at each position
  // input: probs are (B,T,Vp) of the probabilities
  // input: targets is (B,T) of integers giving the correct index in logits
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // loss = -log(probs[target])
      float* probs_bt = probs + b * T * Vp + t * Vp;
      int ix = targets[b * T + t];
      losses[b * T + t] = -logf(probs_bt[ix]);
    }
  }
}
void crossentropy_backward(float* dprobs, const float* dlosses,
                           const float* probs, const int* targets, int B, int T,
                           int Vp) {
  // backwards through crossentropy
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      float* dprobs_bt = dprobs + b * T * Vp + t * Vp;
      const float* probs_bt = probs + b * T * Vp + t * Vp;
      int ix = targets[b * T + t];
      float dloss = dlosses[b * T + t];
      // the gradient of cross-entropy loss w.r.t. the probabilities is:
      // dL/dprobs[i] = -1/probs[i] for i=target, and 0 otherwise
      // so we get this gradient, scaled by dloss
      dprobs_bt[ix] += -dloss / probs_bt[ix];
    }
  }
}

namespace clad {
namespace custom_derivatives {
void matmul_forward_pullback(float* out, const float* inp, const float* weight,
                             const float* bias, int B, int T, int C, int OC,
                             float* dout, float* dinp, float* dweight,
                             float* dbias, int* dB, int* dT, int* dC,
                             int* dOC) {
  matmul_backward(dinp, dweight, dbias, dout, inp, weight, B, T, C, OC);
}
void encoder_forward_pullback(float* out, const int* inp, float* wte,
                              float* wpe, int B, int T, int C, float* dout,
                              float* dwte, float* dwpe, int* dB, int* dT,
                              int* dC) {
  encoder_backward(dwte, dwpe, dout, inp, B, T, C);
}
void layernorm_forward_pullback(float* out, float* mean, float* rstd,
                                float* inp, float* weight, float* bias, int B,
                                int T, int C, float* dout, float* dmean,
                                float* drstd, float* dinp, float* dweight,
                                float* dbias, int* dB, int* dT, int* dC) {
  layernorm_backward(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T,
                     C);
}
void attention_forward_pullback(float* out, float* preatt, float* att,
                                float* inp, int B, int T, int C, int NH,
                                float* dout, float* dpreatt, float* datt,
                                float* dinp, int* dB, int* dT, int* dC,
                                int* dNH) {
  attention_backward(dinp, dpreatt, datt, dout, inp, att, B, T, C, NH);
}
void residual_forward_pullback(float* out, float* inp1, float* inp2, int N,
                               float* dout, float* dinp1, float* dinp2,
                               int* dN) {
  residual_backward(dinp1, dinp2, dout, N);
}
void softmax_forward_pullback(float* probs, float* logits, int B, int T, int V,
                              int Vp, float* dprobs, float* dlogits, int* dB,
                              int* dT, int* dV, int* dVp) {
  softmax_backward(dlogits, dprobs, probs, B, T, V, Vp);
}
void crossentropy_forward_pullback(float* losses, float* probs,
                                   const int* targets, int B, int T, int Vp,
                                   float* dlosses, float* dprobs, int* dB,
                                   int* dT, int* dVp) {
  crossentropy_backward(dprobs, dlosses, probs, targets, B, T, Vp);
}
void gelu_forward_pullback(float* out, const float* inp, int N, float* dout,
                           float* dinp, int* dN) {
  gelu_backward(dinp, inp, dout, N);
}
} // namespace custom_derivatives
} // namespace clad

// ----------------------------------------------------------------------------
// GPT-2 model definition

struct GPT2Config {
  int max_seq_len;       // max sequence length, e.g. 1024
  int vocab_size;        // vocab size, e.g. 50257
  int padded_vocab_size; // padded to e.g. %128==0, 50304
  int num_layers;        // number of layers, e.g. 12
  int num_heads;         // number of heads in attention, e.g. 12
  int channels;          // number of channels, e.g. 768
  GPT2Config() = default;
  GPT2Config(const char* checkpoint_path) {
    // read in model from a checkpoint file
    // gpt2::utils::fread_check(void *ptr, size_t size, size_t nmemb, FILE
    // *stream)
    FILE* model_file = gpt2::utils::fopen_check(checkpoint_path, "rb");
    int model_header[256];
    gpt2::utils::fread_check(model_header, sizeof(int), 256, model_file);
    fclose(model_file);
    if (model_header[0] != 20240326) {
      fprintf(stderr, "Bad magic model file\n");
      exit(1);
    }
    if (model_header[1] != 3) {
      fprintf(stderr, "Bad version in model file\n");
      fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n");
      exit(1);
    }
    // read in hyperparameters
    max_seq_len = model_header[2];
    vocab_size = model_header[3];
    num_layers = model_header[4];
    num_heads = model_header[5];
    channels = model_header[6];
    padded_vocab_size = model_header[7];
  }
};

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
struct ParameterTensors {
  float* wte;      // (V, C)
  float* wpe;      // (maxT, C)
  float* ln1w;     // (L, C)
  float* ln1b;     // (L, C)
  float* qkvw;     // (L, 3*C, C)
  float* qkvb;     // (L, 3*C)
  float* attprojw; // (L, C, C)
  float* attprojb; // (L, C)
  float* ln2w;     // (L, C)
  float* ln2b;     // (L, C)
  float* fcw;      // (L, 4*C, C)
  float* fcb;      // (L, 4*C)
  float* fcprojw;  // (L, C, 4*C)
  float* fcprojb;  // (L, C)
  float* lnfw;     // (C)
  float* lnfb;     // (C)

  float* memory;
  size_t sizes[NUM_PARAMETER_TENSORS];

  // allocate memory for the parameters and point the individual tensors to the
  // right places
  ParameterTensors(GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    sizes[0] = Vp * C;           // wte
    sizes[1] = maxT * C;         // wpe
    sizes[2] = L * C;            // ln1w
    sizes[3] = L * C;            // ln1b
    sizes[4] = L * (3 * C) * C;  // qkvw
    sizes[5] = L * (3 * C);      // qkvb
    sizes[6] = L * C * C;        // attprojw
    sizes[7] = L * C;            // attprojb
    sizes[8] = L * C;            // ln2w
    sizes[9] = L * C;            // ln2b
    sizes[10] = L * (4 * C) * C; // fcw
    sizes[11] = L * (4 * C);     // fcb
    sizes[12] = L * C * (4 * C); // fcprojw
    sizes[13] = L * C;           // fcprojb
    sizes[14] = C;               // lnfw
    sizes[15] = C;               // lnfb
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++)
      num_parameters += sizes[i];
    // malloc all parameters all at once
    memory = new float[num_parameters];
    // assign all the tensors
    float** ptrs[] = {&wte,      &wpe,      &ln1w, &ln1b, &qkvw, &qkvb,
                      &attprojw, &attprojb, &ln2w, &ln2b, &fcw,  &fcb,
                      &fcprojw,  &fcprojb,  &lnfw, &lnfb};
    float* params_memory_iterator = memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
      *(ptrs[i]) = params_memory_iterator;
      params_memory_iterator += sizes[i];
    }
  }
  ~ParameterTensors() { delete[] memory; }
};

#define NUM_ACTIVATION_TENSORS 23
struct ActivationTensors {
  float* encoded;   // (B, T, C)
  float* ln1;       // (L, B, T, C)
  float* ln1_mean;  // (L, B, T)
  float* ln1_rstd;  // (L, B, T)
  float* qkv;       // (L, B, T, 3*C)
  float* atty;      // (L, B, T, C)
  float* preatt;    // (L, B, NH, T, T)
  float* att;       // (L, B, NH, T, T)
  float* attproj;   // (L, B, T, C)
  float* residual2; // (L, B, T, C)
  float* ln2;       // (L, B, T, C)
  float* ln2_mean;  // (L, B, T)
  float* ln2_rstd;  // (L, B, T)
  float* fch;       // (L, B, T, 4*C)
  float* fch_gelu;  // (L, B, T, 4*C)
  float* fcproj;    // (L, B, T, C)
  float* residual3; // (L, B, T, C)
  float* lnf;       // (B, T, C)
  float* lnf_mean;  // (B, T)
  float* lnf_rstd;  // (B, T)
  float* logits;    // (B, T, V)
  float* probs;     // (B, T, V)
  float* losses;    // (B, T)

  float* memory;
  size_t sizes[NUM_ACTIVATION_TENSORS];
  ActivationTensors() : memory(nullptr) {}
  void fill_in_activation_sizes(GPT2Config config, int B, int T) {
    size_t C = config.channels;
    size_t NH = config.num_heads;
    size_t L = config.num_layers;
    size_t Vp = config.padded_vocab_size;
    sizes[0] = B * T * C;          // encoded
    sizes[1] = L * B * T * C;      // ln1
    sizes[2] = L * B * T;          // ln1_mean
    sizes[3] = L * B * T;          // ln1_rstd
    sizes[4] = L * B * T * 3 * C;  // qkv
    sizes[5] = L * B * T * C;      // atty
    sizes[6] = L * B * NH * T * T; // preatt
    sizes[7] = L * B * NH * T * T; // att
    sizes[8] = L * B * T * C;      // attproj
    sizes[9] = L * B * T * C;      // residual2
    sizes[10] = L * B * T * C;     // ln2
    sizes[11] = L * B * T;         // ln2_mean
    sizes[12] = L * B * T;         // ln2_rstd
    sizes[13] = L * B * T * 4 * C; // fch
    sizes[14] = L * B * T * 4 * C; // fch_gelu
    sizes[15] = L * B * T * C;     // fcproj
    sizes[16] = L * B * T * C;     // residual3
    sizes[17] = B * T * C;         // lnf
    sizes[18] = B * T;             // lnf_mean
    sizes[19] = B * T;             // lnf_rstd
    sizes[20] = B * T * Vp;        // logits
    sizes[21] = B * T * Vp;        // probs
    sizes[22] = B * T;             // losses
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++)
      num_activations += sizes[i];
    memory = new float[num_activations];
    float** ptrs[] = {&encoded, &ln1,       &ln1_mean, &ln1_rstd, &qkv,
                      &atty,    &preatt,    &att,      &attproj,  &residual2,
                      &ln2,     &ln2_mean,  &ln2_rstd, &fch,      &fch_gelu,
                      &fcproj,  &residual3, &lnf,      &lnf_mean, &lnf_rstd,
                      &logits,  &probs,     &losses};
    float* acts_memory_iterator = memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
      *(ptrs[i]) = acts_memory_iterator;
      acts_memory_iterator += sizes[i];
    }
  }

  ~ActivationTensors() { delete[] memory; }
};

struct GPT2 {
  GPT2Config config;
  // the weights (parameters) of the model, and their sizes
  ParameterTensors params;
  size_t num_parameters;
  // buffers for the AdamW optimizer
  // float* m_memory;
  // float* v_memory;
  // the activations of the model, and their sizes
  ActivationTensors acts;
  size_t num_activations;
  // other run state configuration
  int batch_size;  // the batch size (B) of current forward pass
  int seq_len;     // the sequence length (T) of current forward pass
  float mean_loss; // after a forward pass with targets, will be populated with
                   // the mean loss

  GPT2(const char* checkpoint_path)
      : config(checkpoint_path), params(config), num_parameters(0) {
    FILE* model_file = gpt2::utils::fopen_check(checkpoint_path, "rb");
    int model_header[256];
    gpt2::utils::fread_check(model_header, sizeof(int), 256, model_file);
    // count the number of parameters
    for (size_t size : this->params.sizes)
      num_parameters += size;
    // read in all the parameters from file
    gpt2::utils::fread_check(this->params.memory, sizeof(float), num_parameters,
                             model_file);
    fclose(model_file);

    // other inits
    this->batch_size = 0;
    this->seq_len = 0;
    this->mean_loss = -1.0f; // -1.0f will designate no loss
  }

  GPT2(GPT2Config config) : config(config), params(config), num_parameters(0) {
    // count the number of parameters
    for (size_t size : this->params.sizes)
      num_parameters += size;

    // other inits
    this->batch_size = 0;
    this->seq_len = 0;
    this->mean_loss = -1.0f; // -1.0f will designate no loss
  }

  void allocate(size_t B, size_t T) {
    if (acts.memory)
      return;
    // record the current B,T as well
    batch_size = B;
    seq_len = T;
    // and now allocate the space
    acts.fill_in_activation_sizes(config, B, T);
    num_activations = 0;
    for (unsigned long size : acts.sizes)
      num_activations += size;
  }

  void zero_all() {
    mean_loss = 0;
    std::fill_n(params.memory, num_parameters, 0.0f);
    if (acts.memory)
      std::fill_n(acts.memory, num_activations, 0.0f);
  }

  void update(const GPT2* d_model, const float lr) {
#pragma omp simd
    for (int i = 0; i < num_parameters; i++)
      params.memory[i] -= lr * d_model->params.memory[i];
  }

  void forward(const int* inputs, const int* targets) {
    size_t V = config.vocab_size;
    size_t Vp = config.padded_vocab_size;
    size_t L = config.num_layers;
    size_t NH = config.num_heads;
    size_t C = config.channels;
    size_t B = batch_size;
    size_t T = seq_len;

    // validate inputs, all indices must be in the range [0, V)
    // for (int i = 0; i < B * T; i++) {
    //   assert(0 <= inputs[i] && inputs[i] < V);
    //   if (targets != nullptr)
    //     assert(0 <= targets[i] && targets[i] < V);
    // }

    // forward pass
    float* residual = acts.encoded;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T,
                    C); // encoding goes into residual[0]
    for (int l = 0; l < L; l++) {
      if (l > 0)
        residual = acts.residual3 + (l - 1) * B * T * C;

      // get the pointers of the weights for this layer
      float* l_ln1w = params.ln1w + l * C;
      float* l_ln1b = params.ln1b + l * C;
      float* l_qkvw = params.qkvw + l * 3 * C * C;
      float* l_qkvb = params.qkvb + l * 3 * C;
      float* l_attprojw = params.attprojw + l * C * C;
      float* l_attprojb = params.attprojb + l * C;
      float* l_ln2w = params.ln2w + l * C;
      float* l_ln2b = params.ln2b + l * C;
      float* l_fcw = params.fcw + l * 4 * C * C;
      float* l_fcb = params.fcb + l * 4 * C;
      float* l_fcprojw = params.fcprojw + l * C * 4 * C;
      float* l_fcprojb = params.fcprojb + l * C;

      // get the pointers of the activations for this layer
      float* l_ln1 = acts.ln1 + l * B * T * C;
      float* l_ln1_mean = acts.ln1_mean + l * B * T;
      float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
      float* l_qkv = acts.qkv + l * B * T * 3 * C;
      float* l_atty = acts.atty + l * B * T * C;
      float* l_preatt = acts.preatt + l * B * NH * T * T;
      float* l_att = acts.att + l * B * NH * T * T;
      float* l_attproj = acts.attproj + l * B * T * C;
      float* l_residual2 = acts.residual2 + l * B * T * C;
      float* l_ln2 = acts.ln2 + l * B * T * C;
      float* l_ln2_mean = acts.ln2_mean + l * B * T;
      float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
      float* l_fch = acts.fch + l * B * T * 4 * C;
      float* l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C;
      float* l_fcproj = acts.fcproj + l * B * T * C;
      float* l_residual3 = acts.residual3 + l * B * T * C;

      // now do the forward pass
      layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b,
                        B, T, C);
      matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
      attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
      matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
      residual_forward(l_residual2, residual, l_attproj, B * T * C);
      layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w,
                        l_ln2b, B, T, C);
      matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
      gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
      matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C,
                     C);
      residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
    }
    residual =
        acts.residual3 + (L - 1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual,
                      params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, nullptr, B, T, C, Vp);
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

    // also forward the cross-entropy loss function if we have the targets
    if (targets != nullptr) {
      crossentropy_forward(acts.losses, acts.probs, targets, B, T, Vp);
      // for convenience also evaluate the mean loss
      mean_loss = 0.0f;
      for (int i = 0; i < B * T; i++)
        mean_loss += acts.losses[i];
      mean_loss /= B * T;
    } else {
      // if we don't have targets, we don't have a loss
      mean_loss = -1.0f;
    }
  }
};

// void gpt2_update(GPT2* model, float learning_rate, float beta1, float beta2,
// float eps, float weight_decay, int t) {
//   // reference:
//   https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

//   // lazily allocate the memory for m_memory and v_memory
//   if (model->m_memory == nullptr) {
//     model->m_memory = (float*)calloc(model->num_parameters, sizeof(float));
//     model->v_memory = (float*)calloc(model->num_parameters, sizeof(float));
//   }

//   for (size_t i = 0; i < model->num_parameters; i++) {
//     float param = model->params.memory[i];
//     float grad = model->grads_memory[i];

//     // update the first moment (momentum)
//     float m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
//     // update the second moment (RMSprop)
//     float v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
//     // bias-correct both moments
//     float m_hat = m / (1.0f - powf(beta1, t));
//     float v_hat = v / (1.0f - powf(beta2, t));

//     // update
//     model->m_memory[i] = m;
//     model->v_memory[i] = v;
//     model->params.memory[i] -= learning_rate * (m_hat / (std::sqrt(v_hat) +
//     eps) + weight_decay * param);
//   }
// }
