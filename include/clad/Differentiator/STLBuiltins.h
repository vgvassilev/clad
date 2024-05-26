#ifndef CLAD_STL_BUILTINS_H
#define CLAD_STL_BUILTINS_H

#include <clad/Differentiator/BuiltinDerivatives.h>
#include <initializer_list>
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
ValueAndPushforward<typename ::std::initializer_list<T>::iterator,
                    typename ::std::initializer_list<T>::iterator>
begin_pushforward(::std::initializer_list<T>* il,
                  ::std::initializer_list<T>* d_il) {
  return {il->begin(), d_il->begin()};
}

template <typename T>
ValueAndPushforward<typename ::std::initializer_list<T>::iterator,
                    typename ::std::initializer_list<T>::iterator>
end_pushforward(const ::std::initializer_list<T>* il,
                const ::std::initializer_list<T>* d_il) {
  return {il->end(), d_il->end()};
}

template <typename T>
clad::ValueAndPushforward<::std::vector<T>, ::std::vector<T>>
constructor_pushforward(
    ConstructorPushforwardTag<::std::vector<T>>,
    typename ::std::vector<T>::size_type n,
    const typename ::std::vector<T>::allocator_type& alloc,
    typename ::std::vector<T>::size_type d_n,
    const typename ::std::vector<T>::allocator_type& d_alloc) {
  ::std::vector<T> v(n, alloc);
  ::std::vector<T> d_v(n, 0, alloc);
  return {v, d_v};
}

template <typename T>
clad::ValueAndPushforward<::std::vector<T>, ::std::vector<T>>
constructor_pushforward(
    ConstructorPushforwardTag<::std::vector<T>>,
    typename ::std::vector<T>::size_type n, T val,
    const typename ::std::vector<T>::allocator_type& alloc,
    typename ::std::vector<T>::size_type d_n, T d_val,
    const typename ::std::vector<T>::allocator_type& d_alloc) {
  ::std::vector<T> v(n, val, alloc);
  ::std::vector<T> d_v(n, d_val, alloc);
  return {v, d_v};
}

template <typename T>
clad::ValueAndPushforward<::std::vector<T>, ::std::vector<T>>
constructor_pushforward(
    ConstructorPushforwardTag<::std::vector<T>>,
    ::std::initializer_list<T> list,
    const typename ::std::vector<T>::allocator_type& alloc,
    ::std::initializer_list<T> dlist,
    const typename ::std::vector<T>::allocator_type& dalloc) {
  ::std::vector<T> v(list, alloc);
  ::std::vector<T> d_v(dlist, dalloc);
  return {v, d_v};
}

template <typename T>
clad::ValueAndPushforward<::std::vector<T>, ::std::vector<T>>
constructor_pushforward(ConstructorPushforwardTag<::std::vector<T>>,
                        typename ::std::vector<T>::size_type n,
                        typename ::std::vector<T>::size_type d_n) {
  ::std::vector<T> v(n);
  ::std::vector<T> d_v(n, 0);
  return {v, d_v};
}

template <typename T>
clad::ValueAndPushforward<::std::vector<T>, ::std::vector<T>>
constructor_pushforward(ConstructorPushforwardTag<::std::vector<T>>,
                        typename ::std::vector<T>::size_type n, T val,
                        typename ::std::vector<T>::size_type d_n, T d_val) {
  ::std::vector<T> v(n, val);
  ::std::vector<T> d_v(n, d_val);
  return {v, d_v};
}

template <typename T>
clad::ValueAndPushforward<::std::vector<T>, ::std::vector<T>>
constructor_pushforward(ConstructorPushforwardTag<::std::vector<T>>,
                        ::std::initializer_list<T> list,
                        ::std::initializer_list<T> dlist) {
  ::std::vector<T> v(list);
  ::std::vector<T> d_v(dlist);
  return {v, d_v};
}

template <typename T>
ValueAndPushforward<T&, T&>
operator_subscript_pushforward(::std::vector<T>* v, unsigned idx,
                               ::std::vector<T>* d_v, unsigned d_idx) {
  return {(*v)[idx], (*d_v)[idx]};
}

template <typename T>
ValueAndPushforward<const T&, const T&>
operator_subscript_pushforward(const ::std::vector<T>* v, unsigned idx,
                               const ::std::vector<T>* d_v, unsigned d_idx) {
  return {(*v)[idx], (*d_v)[idx]};
}

} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad

#endif // CLAD_STL_BUILTINS_H
