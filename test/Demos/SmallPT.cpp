// A path tracer that gets its surface normals from the derivative of a
// distance function. Rendering an image takes minutes, so this stops at
// compiling it.
//
// RUN: %cladclang %S/../../demos/ComputerGraphics/smallpt/SmallPT.cpp -I%S/../../include -o%t
