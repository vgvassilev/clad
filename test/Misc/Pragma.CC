// RUN: %cladclang -fsyntax-only -Xclang -verify %s

#pragma clad DEFAULT // FIXME: Moving DEFAULT after ON/OFF makes the test crash.
#pragma clad ON
#pragma clad OFF

#pragma clad AAA // expected-warning {{expected 'ON' or 'OFF' or 'DEFAULT' in pragma}}
#pragma clang diagnostic clad // expected-warning {{pragma diagnostic expected 'error', 'warning', 'ignored', 'fatal', 'push', or 'pop'}}

// FIXME: Enumerate the various scenarios of decls and clad:: calls between
// on/off/default regions
