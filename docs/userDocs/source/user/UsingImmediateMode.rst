Using Clad-generated derivatives in an immediate context
**********************************************************

The derivatives that Clad generates are valid C++ code, which could in theory
be executed at compile-time (or in an immediate context as the C++ standard
calls it). When a function is differentiated all specifiers, such as
`constexpr` and `consteval` are kept, but it is important to understand the
interface that Clad provides for those derivatives to the user.

When Clad differentiates a function (e.g. with `clad::differentiate`) the user
receives a `CladFunction`, which contains a function pointer to the generated
derivative, among many other things. Unfortunately due to how the C++ standard
is written handling function pointers in an immediate context is very
restricted and care needs to be taken to not violate the rules or the compiler
won't be able to evaluate our `constexpr`/`consteval` functions during
translation.

Currently to get a `CladFunction` that is usable in immediate mode the user has
to pass `clad::immediate_mode` to the differentiation function and that removes
the ability to dump the generated derivative, but it may be possible to add
support for that in the future.

Usage of Clad's immediate mode
================================================

The following code snippet shows how one can request Clad to use the immediate
mode for differentiation::

    #include "clad/Differentiator/Differentiator.h"

    constexpr double fn(double x, double y) {
        return (x + y) / 2;
    }

    constexpr double fn_test() {
        auto dx = clad::differentiate<clad::immediate_mode>(fn, "x");

        return dx.execute(4, 7);
    }

    int main(){
        constexpr double fn_result = fn_test();

        printf("%.2f\n", fn_result);
    }

It is neccessary both to pass the `clad::immediate_mode` option to
`clad::differentiate` and to keep both the call to `clad::differentiate` and
all it's `.execute(...)` calls in the same immediate context, as the C++
standard forbids having a function pointer to an immediate function outside of
an immediate context. (It is not possible to do the differentiation and
executions in main as `dx` would contain such a pointer, but `main` is not and
can not be immediate)

When using `constexpr` there is no easy way to tell whether the functions are
actually being evaluated during translation, so it is a good idea to use either
`consteval` or an `if consteval` (in C++23 and newer) to check if the immediate
contexts are behaving as expected or assign the results to a variable marked
`constexpr` as that would fail if the expression that is being assigned isn't
immediate.

Use cases supported by Clad's immediate mode
================================================

Currently Clad's immediate mode is primarily meant to be used in the forward
mode (`clad::differentiate`) as internal data structures that Clad needs for
differentiating loops, etc. are not yet usable in an immediate context.

Both `constexpr` and `consteval` are supported as Clad doesn't actually rely on
these specific keywords for its support, but instead uses clang's API to
determine if the functions are immediate and should be differentiated eariler.
