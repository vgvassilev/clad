#ifndef CLAD_STL_BUILTINS_H
#define CLAD_STL_BUILTINS_H

#include <vector>
#include "clad/Differentiator/BuiltinDerivatives.h"

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
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad

#endif // CLAD_STL_BUILTINS_H
