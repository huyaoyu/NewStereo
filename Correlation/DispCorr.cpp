#include <torch/extension.h>

#include <iostream>
#include <vector>

std::vector<torch::Tensor> disp_corr_forward( torch::Tensor input )
{
    return { torch::sigmoid( input ) };
}

std::vector<torch::Tensor> disp_corr_backward( torch::Tensor grad, torch::Tensor s, torch::Tensor x )
{
    auto enx = torch::exp( -1.0 * x );
    auto sp  = enx * s * s;

    return { grad * sp };
}

PYBIND11_MODULE( TORCH_EXTENSION_NAME, m )
{
    m.def("forward", &disp_corr_forward, "DispCorr forward");
    m.def("backward", &disp_corr_backward, "DispCorr backward");
}

