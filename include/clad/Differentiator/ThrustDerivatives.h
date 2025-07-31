#ifndef CLAD_DIFFERENTIATOR_THRUSTDERIVATIVES_H
#define CLAD_DIFFERENTIATOR_THRUSTDERIVATIVES_H

#include <cstddef>
#include <iterator>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <type_traits>

#include "clad/Differentiator/Differentiator.h"

namespace clad::custom_derivatives::thrust {

template <typename Iterator, typename T, typename BinaryOp>
void reduce_pullback(Iterator first, Iterator last, T init, BinaryOp op,
                     T d_output, Iterator* d_first, Iterator* d_last, T* d_init,
                     BinaryOp* d_op) {
  size_t n = ::thrust::distance(first, last);

  auto d_first_const_ptr = ::thrust::raw_pointer_cast((*d_first).base());
  auto d_first_ptr = const_cast<T*>(d_first_const_ptr);
  ::thrust::device_ptr<T> d_first_dev_ptr(d_first_ptr);

  if constexpr (::std::is_same_v<BinaryOp, ::thrust::plus<T>>) {
    if (d_init)
      *d_init += d_output;

    struct add_d_output {
      T d_output;
      add_d_output(T d) : d_output(d) {}
      __host__ __device__ void operator()(T& x) const { x += d_output; }
    };

    if (n > 0) {
      ::thrust::for_each(d_first_dev_ptr, d_first_dev_ptr + n,
                         add_d_output(d_output));
    }

  } else if constexpr (::std::is_same_v<BinaryOp, ::thrust::maximum<T>>) {
    auto maxValue = ::thrust::reduce(first, last, init, ::thrust::maximum<T>());
    auto k = (init == maxValue) ? 1 : 0;
    k += ::thrust::count(first, last, maxValue);

    if (k > 0) {
      auto gradient = d_output / k;
      if (d_init && init == maxValue)
        *d_init += gradient;

      struct max_grad_functor {
        T max_val, gradient;
        max_grad_functor(T mv, T g) : max_val(mv), gradient(g) {}
        __host__ __device__ void
        operator()(::thrust::tuple<T&, const T&> t) const {
          if (::thrust::get<1>(t) == max_val)
            ::thrust::get<0>(t) += gradient;
        }
      };

      if (n > 0) {
        auto iter = ::thrust::make_zip_iterator(
            ::thrust::make_tuple(d_first_dev_ptr, first));
        ::thrust::for_each(iter, iter + n,
                           max_grad_functor(maxValue, gradient));
      }
    }

  } else if constexpr (::std::is_same_v<BinaryOp, ::thrust::minimum<T>>) {
    auto minValue = ::thrust::reduce(first, last, init, ::thrust::minimum<T>());
    auto k = (init == minValue) ? 1 : 0;
    k += ::thrust::count(first, last, minValue);

    if (k > 0) {
      auto gradient = d_output / k;
      if (d_init && init == minValue)
        *d_init += gradient;

      struct min_grad_functor {
        T min_val, gradient;
        min_grad_functor(T mv, T g) : min_val(mv), gradient(g) {}
        __host__ __device__ void
        operator()(::thrust::tuple<T&, const T&> t) const {
          if (::thrust::get<1>(t) == min_val)
            ::thrust::get<0>(t) += gradient;
        }
      };

      if (n > 0) {
        auto iter = ::thrust::make_zip_iterator(
            ::thrust::make_tuple(d_first_dev_ptr, first));
        ::thrust::for_each(iter, iter + n,
                           min_grad_functor(minValue, gradient));
      }
    }

  } else if constexpr (::std::is_same_v<BinaryOp, ::thrust::multiplies<T>>) {
    size_t zero_count = ::thrust::count(first, last, 0);
    bool init_is_zero = (init == 0);

    if (zero_count > 1 || (zero_count == 1 && init_is_zero)) {
    } else if (zero_count == 1 && !init_is_zero) {
      struct replace_zero_with_one {
        __host__ __device__ T operator()(const T& x) const {
          return (x == 0) ? 1 : x;
        }
      };
      T non_zero_product = ::thrust::reduce(
          ::thrust::make_transform_iterator(first, replace_zero_with_one()),
          ::thrust::make_transform_iterator(last, replace_zero_with_one()),
          init, ::thrust::multiplies<T>());

      struct single_zero_grad_functor {
        T gradient;
        single_zero_grad_functor(T g) : gradient(g) {}
        __host__ __device__ void
        operator()(::thrust::tuple<T&, const T&> t) const {
          if (::thrust::get<1>(t) == 0)
            ::thrust::get<0>(t) += gradient;
        }
      };
      ::thrust::for_each(::thrust::make_zip_iterator(
                             ::thrust::make_tuple(d_first_dev_ptr, first)),
                         ::thrust::make_zip_iterator(
                             ::thrust::make_tuple(d_first_dev_ptr + n, last)),
                         single_zero_grad_functor(d_output * non_zero_product));

    } else if (zero_count == 0 && init_is_zero) {
      if (d_init) {
        T vector_product =
            ::thrust::reduce(first, last, (T)1, ::thrust::multiplies<T>());
        *d_init += d_output * vector_product;
      }
    } else {
      auto product =
          ::thrust::reduce(first, last, init, ::thrust::multiplies<T>());

      if (d_init)
        *d_init += d_output * product / init;

      struct multiplies_grad_functor {
        T d_output, product;
        multiplies_grad_functor(T d, T p) : d_output(d), product(p) {}
        __host__ __device__ T operator()(const T& d_x, const T& x) const {
          return d_x + d_output * product / x;
        }
      };

      if (n > 0) {
        ::thrust::transform(d_first_dev_ptr, d_first_dev_ptr + n, first,
                            d_first_dev_ptr,
                            multiplies_grad_functor(d_output, product));
      }
    }

  } else if constexpr (::std::is_same_v<BinaryOp, ::thrust::bit_and<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::bit_or<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::bit_xor<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::equal_to<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::greater<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::greater_equal<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::less<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::less_equal<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::logical_and<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::logical_or<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::not_equal_to<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::modulus<T>>) {
  } else {
    static_assert(::std::is_same_v<T, void>,
                  "This binary operation is not supported by the custom "
                  "reduce_pullback.");
  }
}

} // namespace clad::custom_derivatives::thrust

#endif // CLAD_DIFFERENTIATOR_THRUSTDERIVATIVES_H