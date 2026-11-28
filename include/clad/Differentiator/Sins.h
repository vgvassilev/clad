#ifndef CLAD_DIFFERENTIATOR_SINS_H
#define CLAD_DIFFERENTIATOR_SINS_H

#include <type_traits>

/// Standard-protected facility allowing access into private members in C++. Use
/// with caution!
/// Access checks do not apply to the names in an explicit instantiation
/// ([temp.spec.general]). So `template struct member_access<Key, &C::m>;` may
/// name a private `m`. The instantiation defines the friend function
/// member_pointer() that Key declares, which returns `&C::m`; code finds that
/// function through Key.

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
// The friend is meant to be a function, not a function template.
#pragma GCC diagnostic ignored "-Wnon-template-friend"
#endif

namespace clad {
/// The type of a pointer to a member of type \p T in \p Class. \p T may be an
/// array.
template <class T, class Class> using member_pointer_t = T Class::*;

/// Identifies member number \p Id, of type \p T, in \p Class. Instantiating
/// member_access with this key gives access to it.
template <class Class, class T, unsigned Id> struct member_key {
  using type = member_pointer_t<T, Class>;
  friend type member_pointer(member_key*);
};

/// Identifies the member that \p Tag stands for; `Tag::type` is the type of a
/// pointer to it. Used by CLAD_ALLOW_ACCESS, which identifies members by name.
template <class Tag> struct tagged_member_key {
  using type = typename Tag::type;
  friend type member_pointer(tagged_member_key*);
};

/// When explicitly instantiated, defines the member_pointer() that \p Key
/// declares to return \p Member.
template <class Key, typename Key::type Member> struct member_access {
  friend typename Key::type member_pointer(Key*) { return Member; }
};

/// Returns member number \p Id, of type \p T, of \p obj. Needs a member_access
/// instantiation for this key.
template <class Class, class T, unsigned Id> T& member(Class& obj) {
  return obj.*member_pointer((member_key<Class, T, Id>*)nullptr);
}
template <class Class, class T, unsigned Id> const T& member(const Class& obj) {
  return obj.*member_pointer((member_key<Class, T, Id>*)nullptr);
}
} // namespace clad

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

/// Gives CLAD_ACCESS access to \p MEMBER of \p CLASS, whose type is
/// `__VA_ARGS__`. The key's tag is named after the member so that CLAD_ACCESS
/// can refer to it. Use it at global scope or in namespace clad: an explicit
/// instantiation must be in a namespace that encloses its template
/// ([temp.explicit]).
#define CLAD_ALLOW_ACCESS(CLASS, MEMBER, ...)                                  \
  template <typename> struct Only_##MEMBER;                                    \
  template <> struct Only_##MEMBER<CLASS> {                                    \
    using type = ::clad::member_pointer_t<__VA_ARGS__, CLASS>;                 \
  };                                                                           \
  template struct ::clad::member_access<                                       \
      ::clad::tagged_member_key<Only_##MEMBER<CLASS>>, &CLASS::MEMBER>

// Parenthesized as a whole: `.*` binds looser than the `[]` or `()` that may
// follow. The object's const is dropped with its reference, as Only_ is
// specialized for the class itself.
#define CLAD_ACCESS(OBJECT, MEMBER)                                            \
  ((OBJECT).*                                                                  \
   member_pointer((::clad::tagged_member_key<Only_##MEMBER<                    \
                       typename std::remove_cv<typename std::remove_reference< \
                           decltype(OBJECT)>::type>::type>>*)nullptr))

#endif // CLAD_DIFFERENTIATOR_SINS_H
