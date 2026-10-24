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

#pragma clad AAA // expected-error {{expected 'ON', 'OFF', 'DEFAULT', or `checkpoint` in pragma}}
#pragma clad 123 // expected-error {{expected 'ON', 'OFF', 'DEFAULT', or `checkpoint` in pragma}}
#pragma clad     // expected-error {{expected 'ON', 'OFF', 'DEFAULT', or `checkpoint` in pragma}}
#pragma clad checkpoint // expected-error {{expected 'loop' after 'checkpoint' in #pragma clad}}
#pragma clad checkpoint 123 // expected-error {{expected 'loop' after 'checkpoint' in #pragma clad}}
#pragma clang diagnostic clad // expected-warning {{pragma diagnostic expected 'error', 'warning', 'ignored', 'fatal', 'push', or 'pop'}}

// Testing Issue #2320: ON/OFF macros must not affect #pragma clad ON/OFF
#define ON 1
#define OFF 2
#pragma clad ON
#pragma clad OFF
