#ifndef CLAD_STL_BUILTINS_H
#define CLAD_STL_BUILTINS_H

#include <clad/Differentiator/BuiltinDerivatives.h>
#include <vector>

namespace clad {
namespace custom_derivatives {
namespace class_functions {

template <typename T>
void clear_pushforward(::std::vector<T>* v, ::std::vector<T>* d_v) {
  d_v->clear();
  v->clear();
}

template <typename T>
void resize_pushforward(::std::vector<T>* v, unsigned sz, ::std::vector<T>* d_v,
                        unsigned d_sz) {
  d_v->resize(sz, T());
  v->resize(sz);
}

template <typename T, typename U>
void resize_pushforward(::std::vector<T>* v, unsigned sz, U val,
                        ::std::vector<T>* d_v, unsigned d_sz, U d_val) {
  d_v->resize(sz, d_val);
  v->resize(sz, val);
}

template <typename T>
clad::ValueAndPushforward<::std::vector<T>, ::std::vector<T>>
constructor_pushforward(Identify<::std::vector<T>>, size_t n,
                        const typename ::std::vector<T>::allocator_type alloc,
                        size_t d_n,
                        const typename ::std::vector<T>::allocator_type d_alloc) {
  ::std::vector<T> v(n, alloc);
  ::std::vector<T> d_v(n, 0, alloc);
  return {v, d_v};
}

template <typename T>
clad::ValueAndPushforward<::std::vector<T>, ::std::vector<T>>
constructor_pushforward(Identify<::std::vector<T>>, size_t n, T val,
                        const typename ::std::vector<T>::allocator_type alloc,
                        size_t d_n, T d_val,
                        const typename ::std::vector<T>::allocator_type d_alloc) {
  ::std::vector<T> v(n, val, alloc);
  ::std::vector<T> d_v(n, d_val, alloc);
  return {v, d_v};
}

template <typename T>
ValueAndPushforward<T&, T&>
operator_subscript_pushforward(::std::vector<T>* v, int idx,
                               ::std::vector<T>* d_v, int d_idx) {
  return {(*v)[idx], (*d_v)[idx]};
}

} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad

#endif // CLAD_STL_BUILTINS_H
