#ifndef CLAD_DIFFERENTIATOR_THRUSTBUILTINS_H
#define CLAD_DIFFERENTIATOR_THRUSTBUILTINS_H

// NOLINTBEGIN(misc-include-cleaner)
#include <thrust/device_vector.h>
#include <thrust/fill.h>
// NOLINTEND(misc-include-cleaner)

namespace clad {

#define elidable_reverse_forw __attribute__((annotate("elidable_reverse_forw")))

// zero_init for thrust::device_vector
template <class T> inline void zero_init(::thrust::device_vector<T>& v) {
  ::thrust::fill(v.begin(), v.end(), T(0)); // NOLINT(misc-include-cleaner)
}

namespace custom_derivatives::class_functions {

// Constructors (reverse mode pullbacks)

template <typename T>
clad::ValueAndAdjoint<::thrust::device_vector<T>, ::thrust::device_vector<T>>
constructor_reverse_forw(clad::Tag<::thrust::device_vector<T>>,
                         typename ::thrust::device_vector<T>::size_type count,
                         const T& val,
                         typename ::thrust::device_vector<T>::size_type d_count,
                         const T& d_val) elidable_reverse_forw;

template <typename T>
inline void
constructor_pullback(typename ::thrust::device_vector<T>::size_type count,
                     const T& val, ::thrust::device_vector<T>* d_this,
                     typename ::thrust::device_vector<T>::size_type* d_count,
                     T* d_val) {
  for (typename ::thrust::device_vector<T>::size_type i = 0; i < count; ++i) {
    *d_val += (*d_this)[i];
    (*d_this)[i] = T(0);
  }
}

template <typename T>
clad::ValueAndAdjoint<::thrust::device_vector<T>, ::thrust::device_vector<T>>
constructor_reverse_forw(clad::Tag<::thrust::device_vector<T>>,
                         typename ::thrust::device_vector<T>::size_type count,
                         typename ::thrust::device_vector<T>::size_type d_count)
    elidable_reverse_forw;

template <typename T>
inline void
constructor_pullback(typename ::thrust::device_vector<T>::size_type count,
                     ::thrust::device_vector<T>* d_this,
                     typename ::thrust::device_vector<T>::size_type* d_count) {
  for (typename ::thrust::device_vector<T>::size_type i = 0; i < count; ++i)
    (*d_this)[i] = T(0);
}

} // namespace custom_derivatives::class_functions

} // namespace clad

#endif // CLAD_DIFFERENTIATOR_THRUSTBUILTINS_H
