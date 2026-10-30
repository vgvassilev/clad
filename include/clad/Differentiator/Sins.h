#ifndef CLAD_DIFFERENTIATOR_SINS_H
#define CLAD_DIFFERENTIATOR_SINS_H

#include <type_traits>

/// Standard-protected facility allowing access into private members in C++.
/// Use with caution!
#define CONCATE_(X, Y) X##Y
#define CONCATE(X, Y) CONCATE_(X, Y)
// The helper struct is named apart with __COUNTER__, not __LINE__: two
// headers can open members of one name on the same line of each.
#define ALLOW_ACCESS(CLASS, MEMBER, ...)                                       \
  CLAD_ALLOW_ACCESS_IMPL(CONCATE(MEMBER, __COUNTER__), CLASS, MEMBER,          \
                         __VA_ARGS__)
#define CLAD_ALLOW_ACCESS_IMPL(NAME, CLASS, MEMBER, ...)                       \
  template <typename Only, __VA_ARGS__ CLASS::*Member> struct NAME {           \
    friend __VA_ARGS__ CLASS::*Access(Only*) { return Member; }                \
  };                                                                           \
  template <typename> struct Only_##MEMBER;                                    \
  template <> struct Only_##MEMBER<CLASS> {                                    \
    friend __VA_ARGS__ CLASS::*Access(Only_##MEMBER<CLASS>*);                  \
  };                                                                           \
  template struct NAME<Only_##MEMBER<CLASS>, &CLASS::MEMBER>

// Parenthesized as a whole: `.*` binds looser than the `[]` or `()` that may
// follow. The object's const is dropped with its reference, as Only_ is
// specialized for the class itself.
#define ACCESS(OBJECT, MEMBER)                                                 \
  ((OBJECT).*                                                                  \
   Access(                                                                     \
       (Only_##MEMBER<typename std::remove_cv<typename std::remove_reference<  \
            decltype(OBJECT)>::type>::type>*)nullptr))

#endif // CLAD_DIFFERENTIATOR_SINS_H
