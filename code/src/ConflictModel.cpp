#include "ConflictModel.hpp"
#include <torch/torch.h>


ConfNetImpl::ConfNetImpl(int64_t map_width, int64_t map_height, int64_t fc_hidden, int64_t num_output){
    register_module("ConvLayers", ConvLayers);
}

torch::Tensor ConfNetImpl::forward(torch::Tensor x) {
    return ConvLayers->forward(x);
}