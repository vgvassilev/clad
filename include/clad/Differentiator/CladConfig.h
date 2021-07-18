// This file defines configuration that is shared by various files

#ifndef CLAD_CONFIG_H
#define CLAD_CONFIG_H

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

#endif // CLAD_CONFIG_H
