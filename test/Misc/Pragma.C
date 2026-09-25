// RUN: %cladclang -fsyntax-only -Xclang -verify %s

#pragma clad DEFAULT
#pragma clad ON
#pragma clad OFF

// A second OFF/DEFAULT while clad is already off is ignored: it must neither
// assert nor extend the already closed interval.
#pragma clad OFF
#pragma clad DEFAULT
#pragma clad DEFAULT

// Same for a second ON while clad is already on.
#pragma clad ON
#pragma clad ON

#pragma clad AAA // expected-error {{expected 'ON', 'OFF', 'DEFAULT', 'gradient', 'differentiate', or `checkpoint` in pragma}}
#pragma clang diagnostic clad // expected-warning {{pragma diagnostic expected 'error', 'warning', 'ignored', 'fatal', 'push', or 'pop'}}

// FIXME: Enumerate the various scenarios of decls and clad:: calls between
// on/off/default regions
