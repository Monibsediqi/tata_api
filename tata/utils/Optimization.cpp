#include "Optimization.h"

void setInferenceMode(){
	torch::jit::setGraphExecutorOptimize(false);
	torch::jit::getProfilingMode() = false;
	torch::jit::getExecutorMode() = false;
	torch::jit::getBailoutDepth() = 1;
	FLAGS_torch_jit_enable_new_executor = false;
	torch::autograd::AutoGradMode guard(false);
}