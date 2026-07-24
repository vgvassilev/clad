// Companion "library" header for PortingHintsTemplate.cpp: a class template
// whose instantiation is the differentiation boundary, exercising the
// template-argument spelling of the porting hints.
#pragma once

template <class T> struct Boxed {
  T v;
  T scale(T x) const { return x * v; }
};
