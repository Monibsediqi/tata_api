#pragma once
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>


#ifdef DEBUG
//do something
#else

void setInferenceMode();
//do something else



#endif // DEBUG
