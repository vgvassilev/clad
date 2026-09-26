// A fixed number of time steps of particle motion, differentiated with
// respect to the initial velocities. Clad has to carry the derivative
// through the loop that advances the simulation.
//
// RUN: %cladclang_cuda -I%S/../../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -o%t \
// RUN:     %S/../../../demos/CUDA/ParticleSimulation.cu 2>&1 | %filecheck_nodiag %s
//
// Running these needs a device, and what they print is a list of
// gradients rather than anything stable to match, so compiling them
// is the check.
//
// REQUIRES: cuda-compile
