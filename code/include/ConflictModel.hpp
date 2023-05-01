#pragma once
#include <torch/torch.h>

class ConfNetImpl : public torch::nn::Module {
 public:
    ConfNetImpl(int64_t map_width = 10, int64_t map_height = 10, int64_t fc_hidden = 64, int64_t num_output = 1);
    torch::Tensor forward(torch::Tensor x);

 private:
    int64_t map_width;
    int64_t map_height;
    int64_t fc_hidden;
    int64_t num_output;

    torch::nn::Sequential ConvLayers{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 64, 3).stride(1).padding(1)),
        torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
        torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)), //uncomment for version 1
        torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
        torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
        torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
        torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(),
        torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 1, 3).stride(1).padding(1)),
        torch::nn::BatchNorm2d(1),
        torch::nn::Sigmoid()
    };
};

TORCH_MODULE(ConfNet);
