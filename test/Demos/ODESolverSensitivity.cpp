// Clad differentiates the solver itself, loops and all. The demo's output
// is a table of sensitivities with nothing stable to match, so compiling
// it is the check.
//
// RUN: %cladclang %S/../../demos/ODESolverSensitivity.cpp -I%S/../../include -o%t
