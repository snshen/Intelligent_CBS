#pragma once
#include <torch/torch.h>

class ConfNetImpl : public torch::nn::Module {
 public:
    explicit ConfNet(int64_t map_width = 10, int64_t map_height = 10, int64_t fc_hidden = 64, int64_t num_output = 1);
    torch::Tensor forward(torch::Tensor x);

 private:
    torch::nn::Sequential ConvEncoder{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 64, 3).stride(1)),
        torch::nn::BatchNorm2d(16),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),
        torch::nn::BatchNorm2d(16),
        torch::nn::ReLU()
    };

    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};

TORCH_MODULE(ConfNet);
