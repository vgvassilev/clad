// RUN: %cladclang -fsyntax-only -Xclang -verify %s

double add(double a, double b) {
    return a + b;
}

#pragma clad gradient // expected-error {{expected function name in '#pragma clad gradient'}}
#pragma clad differentiate // expected-error {{expected function name in '#pragma clad differentiate'}}

#pragma clad gradient ( // expected-error {{expected function name in '#pragma clad gradient'}}
#pragma clad gradient (add // expected-error {{expected ')' after function name in '#pragma clad gradient'}}

#pragma clad gradient non_existent_fn // expected-error {{cannot find function 'non_existent_fn' for differentiation requested via '#pragma clad'}}
