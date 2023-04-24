#include "ConflictModel.hpp"
#include <torch/torch.h>


ConfNetImpl::ConfNetImpl(int64_t map_width, int64_t map_height, int64_t fc_hidden, int64_t num_output)
    : fc1(map_width * map_height * 64, fc_hidden), fc2(fc_hidden, num_output) {
    register_module("ConvEncoder", ConvEncoder);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
}

torch::Tensor ConfNetImpl::forward(torch::Tensor x) {
    return ConvLayers->forward(x);
}