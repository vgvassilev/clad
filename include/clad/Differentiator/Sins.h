#ifndef CLAD_DIFFERENTIATOR_SINS_H
#define CLAD_DIFFERENTIATOR_SINS_H

#include <type_traits>

/// Standard-protected facility allowing access into private members in C++.
/// Use with caution!
// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define CONCATE_(X, Y) X##Y
#define CONCATE(X, Y) CONCATE_(X, Y)
#define ALLOW_ACCESS(CLASS, MEMBER, ...)                                       \
  template <typename Only, __VA_ARGS__ CLASS::*Member>                         \
  struct CONCATE(MEMBER, __LINE__) {                                           \
    friend __VA_ARGS__ CLASS::*Access(Only*) { return Member; }                \
  };                                                                           \
  template <typename> struct Only_##MEMBER;                                    \
  template <> struct Only_##MEMBER<CLASS> {                                    \
    friend __VA_ARGS__ CLASS::*Access(Only_##MEMBER<CLASS>*);                  \
  };                                                                           \
  template struct CONCATE(MEMBER,                                              \
                          __LINE__)<Only_##MEMBER<CLASS>, &CLASS::MEMBER>

#define ACCESS(OBJECT, MEMBER)                                                 \
  (OBJECT).*Access((Only_##MEMBER<                                             \
                    std::remove_reference<decltype(OBJECT)>::type>*)nullptr)

// NOLINTEND(cppcoreguidelines-macro-usage)

#endif // CLAD_DIFFERENTIATOR_SINS_H
