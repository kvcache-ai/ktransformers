#ifndef CPUINFER_VENDOR_VENDOR_H
#define CPUINFER_VENDOR_VENDOR_H

#ifdef USE_CUDA
#include "cuda.h"
#elif USE_HIP
#define __HIP_PLATFORM_AMD__
#include "hip.h"
#elif USE_MUSA
#include "musa.h"
#endif

#endif  // CPUINFER_VENDOR_VENDOR_H